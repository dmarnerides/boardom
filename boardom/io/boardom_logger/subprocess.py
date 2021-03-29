import traceback
from concurrent.futures import CancelledError
import asyncio
from asyncio import Queue
import aiohttp
import zmq
import json
import boardom as bd
from .serialization import MsgpackAsyncContext
from .subprocess_mixins import (
    _ParentHandlerMixin,
    _ServerHandlerMixin,
    _SubrocessPrivateAPIMixin,
)


class _LoggerSubprocess(
    _ParentHandlerMixin, _ServerHandlerMixin, _SubrocessPrivateAPIMixin
):
    def __init__(self, *args, **kwargs):
        try:
            asyncio.run(self.main(*args, **kwargs))
        except KeyboardInterrupt:
            pass
        except CancelledError:
            pass

    async def main(
        self,
        connected,
        exited,
        child_port,
        parent_port,
        url,
        t_await_reconnect,
        process_id,
        quiet=True,
    ):
        self.quiet = quiet
        try:
            self.should_exit = False
            self.queue_for_ws_tasks = Queue()
            #  ctx = Context.instance()
            ctx = MsgpackAsyncContext.instance()
            self.to_parent = ctx.socket(zmq.PAIR)
            self.to_parent.bind('tcp://*:0')
            sockopt = self.to_parent.getsockopt(zmq.LAST_ENDPOINT)
            child_port.value = int(sockopt.decode('utf-8').split(':')[-1])

            self.from_parent = ctx.socket(zmq.PAIR)
            self.from_parent.connect(f'tcp://127.0.0.1:{parent_port}')
            self.connected = connected
            self.exited = exited
            self.socket_url = f'{url}/engine_socket'
            self.t_await_reconnect = t_await_reconnect
            self.loop = asyncio.get_event_loop()
            #  for s in (SIGHUP, SIGTERM, SIGINT):
            #      self.loop.add_signal_handler(
            #          s, lambda s=s: asyncio.create_task(self._exit(s))
            #      )
            self.ws = None
            self.connection_id = None
            self.process_id = process_id
            persistent_socket_task = asyncio.create_task(self.persist_socket())
            parent_watch_task = asyncio.create_task(self.watch_parent())
            queue_watch_task = asyncio.create_task(self.send_tasks_queued_for_ws())
            try:
                await asyncio.gather(
                    persistent_socket_task, parent_watch_task, queue_watch_task
                )
            except CancelledError:
                self.should_exit = True
                self.write('[Daemon] Subprocess cancelled.')
        except KeyboardInterrupt:
            pass
        except Exception:
            self.write(traceback.format_exc())

        self.write('[Daemon] Process complete.')
        self.exited.value = 1

    def ws_is_alive(self):
        return (
            (self.ws is not None) and (not self.ws.closed) and self.connected.value == 1
        )

    async def persist_socket(self):
        self.loop = asyncio.get_running_loop()
        while True and not self.should_exit:
            try:
                await self.connect_socket()
                await asyncio.sleep(self.t_await_reconnect)
            except CancelledError:
                self.write('[Daemon] Connection cancelled.')
                self.should_exit = True
                break
            except KeyboardInterrupt:
                break
            except Exception:
                self.write(traceback.format_exc())
        self.write('[Daemon] Socket persist function done.')

    async def _close_ws(self):
        if self.ws is not None and not self.ws.closed:
            try:
                await self.ws.close()
            except CancelledError:
                self.should_exit = True
                self.write('[Daemon] Websocket close error.')

            self.connection_id = None

    async def connect_socket(self):
        self.write('[Daemon] Trying to connect to Boardom')
        await self._close_ws()
        try:
            async with aiohttp.ClientSession() as self.session:
                async with self.session.ws_connect(self.socket_url) as self.ws:
                    await self.do_handshake()
                    self.write('[Daemon] Connected to Boardom')
                    self.connected.value = 1
                    try:
                        await self.continuous_receive_from_ws()
                    except CancelledError:
                        self.should_exit = True
                        self.write('[Daemon] WS receiving cancelled.')
                self.write('[Daemon] ws closing')
                await self._close_ws()

        except aiohttp.ClientConnectionError as e:
            self.write(f'[Daemon] ConnectionError: {str(e)}')
            self.connected.value = 0
        except KeyboardInterrupt:
            pass
        except Exception as e:
            self.connected.value = 0
            raise e
        self.connected.value = 0

    async def do_handshake(self):
        self.write('[Daemon] Doing handshake with server (sending process_id)')
        handshake = {
            'payload': {'process_id': self.process_id},
            'type': 'ENGINE_HANDSHAKE',
        }
        try:
            await self.ws.send_json(handshake)
        except CancelledError:
            self.should_exit = True
            self.write('[Daemon] Handshake cancelled')

    async def continuous_receive_from_ws(self):
        async for msg in self.ws:
            try:
                keep_alive = await self._handle_msg_from_server(msg)
            except CancelledError:
                self.should_exit = True
                self.write('[Daemon] Message handler cancelled')
                break
            if not keep_alive:
                self.write('[Daemon] Boardom disconnected')
                break
            if self.should_exit:
                self.write('[Daemon] Main process exiting, disconnecting..')
                return

    async def _handle_msg_from_server(self, msg):
        if msg.type == aiohttp.WSMsgType.TEXT:
            task = json.loads(msg.data)
            request = task['type'].lower()
            try:
                handler = getattr(self, f'server_{request.lower()}')
            except KeyboardInterrupt:
                pass
            except AttributeError:
                handler = self.server_default_handler
            try:
                ret = await handler(task)
            except CancelledError:
                self.should_exit = True
                self.write('[Daemon] Message handler cancelled.')
                return None

            return ret
        elif msg.type == aiohttp.WSMsgType.ERROR:
            self.write(f'[Daemon] Got WebSocket error: {msg.data}')
            return False

    async def watch_parent(self):
        while True and not self.should_exit:
            try:
                try:
                    task = await self.from_parent.recv_msgpack()
                except CancelledError:
                    self.should_exit = True
                    self.write('[Daemon] Cancelled parent receiving coroutine.')
                    break
                except KeyboardInterrupt:
                    break

                self.assign_meta_(task)

                request = task['type'].lower()
                try:
                    handler = getattr(self, f'parent_{request.lower()}')
                except KeyboardInterrupt:
                    break
                except AttributeError:
                    # default handler just puts things in the queue
                    handler = self.parent_default_handler
                try:
                    await handler(task)
                except CancelledError:
                    self.should_exit = True
                    self.write('[Daemon] Task handler cancelled')
                    break
            except KeyboardInterrupt:
                pass
            except Exception:
                self.write(traceback.format_exc())
        self.write('[Daemon] Watch parent function done.')

    async def send_tasks_queued_for_ws(self):
        while True and not self.should_exit:
            try:
                data = await self.queue_for_ws_tasks.get()
            except KeyboardInterrupt:
                self.should_exit = True
                break
            except CancelledError:
                self.should_exit = True
                self.write('[Daemon] Queue for WS tasks cancelled.')
                break
            if data['type'] == 'disconnect':
                self.write('[Daemon] Got disconnect task')
                break
            while True:
                if self.ws_is_alive() and not self.should_exit:
                    try:
                        await self.ws.send_json(data)
                        break
                    except CancelledError:
                        self.should_exit = True
                        self.write('[Daemon] Websocket send cancelled.')
                        break
                    except KeyboardInterrupt:
                        break
                    except RuntimeError:
                        try:
                            await asyncio.sleep(self.t_await_reconnect)
                        except KeyboardInterrupt:
                            break
                        except CancelledError:
                            self.should_exit = True
                            self.write('[Daemon] Websocket send cancelled.')
                            break

                elif self.should_exit:
                    break
                else:
                    try:
                        await asyncio.sleep(self.t_await_reconnect)
                    except CancelledError:
                        self.should_exit = True
                        self.write('[Daemon] Websocket send cancelled.')
                        break
        self.write('[Daemon] Ws task queue function done.')

    def write(self, x):
        if not self.quiet:
            bd.write(x)

    #  async def _exit(self, s):
    #      self.write(f'[Daemon] Got signal {s.name}')
    #      tasks = [
    #          task
    #          for task in asyncio.all_tasks()
    #          if task is not asyncio.current_task() and task.cancel()
    #      ]
    #      await asyncio.gather(*tasks)
    #      self.loop.stop()
