import json
import asyncio
import aiohttp
from aiohttp import web
from .datastore import datastore

# These are received from front end
class FrontMixin:
    async def front_initialize_connection(self):
        await self._front_send_process_list()
        await self._front_send_cfg_store()

    async def front_close(self):
        print(f'Closing front end WS: {self.connection_id}')

    async def front_default_handler(self, task):
        print(f'[Server] (Front) Default handler for {task["type"]}')

    async def request_cfg_store(self, task):
        await self._front_send_cfg_store()

    async def process_list_requested(self, task):
        print('[Server] Front end requested session list)')
        await self._front_send_process_list()

    async def _front_send_process_list(self):
        print('Processes')
        print(list(datastore.store['processes'].keys()))
        await self.send_json(
            {
                'type': 'NEW_PROCESS_LIST_ACQUIRED',
                'payload': datastore.get_all_processes(),
            }
        )

    async def _front_send_cfg_store(self):
        print('[Server] Sending config store')
        await self.send_json(
            {'type': 'ENGINE_CFG_FULL', 'payload': datastore.get_cfg_store()}
        )


# These are received from the boardom logger (subprocess)
class EngineMixin:
    async def engine_handshake(self, task):
        print('[Server] got engine handshake')
        process_id = task['payload']['process_id']
        self.process_id = process_id
        SocketManager.engine_connection_ids[process_id] = self.connection_id
        datastore.add_new_process(process_id)
        # Request for the engine to send the config store
        await self._send('REQUEST_CFG_STORE')
        # Send process info to front end
        await self.broadcast_to_all_fronts(
            {
                'type': 'UPDATE_PROCESS_INFO',
                'payload': datastore.get_process_info(process_id),
            }
        )

    async def engine_initialize_connection(self):
        pass

    async def engine_close(self):
        print(f'Closing engine WS: {self.connection_id}')
        if self.process_id in SocketManager.engine_connection_ids:
            del SocketManager.engine_connection_ids[self.process_id]
        datastore.deactivate_process(self.process_id)
        await self.broadcast_to_all_fronts(
            {'type': 'PROCESS_DEACTIVATED', 'payload': self.process_id}
        )

    async def engine_default_handler(self, task):
        print(
            f'[Server] Default handler for {task["type"]} received from engine. Doing NOTHING!'
        )

    async def engine_session_path(self, task):
        path = task["payload"]
        print(f'[Server] Session path initialized: {path}')
        datastore.store['processes'][self.process_id]['path'] = path
        await self.broadcast_to_all_fronts(task)

    async def engine_cfg_full(self, task):
        print('[Server] Got config store.')
        store = task['payload']
        datastore.add_cfg_store(store, self.process_id)
        # Get the formatted datastore to send
        task['payload'] = datastore.get_cfg_store(self.process_id)
        await self.broadcast_to_all_fronts(task)

    async def set_cfg_value(self, task):
        print('[Server] Setting cfg value')
        task['payload'] = datastore.set_cfg_value(task['payload'], self.process_id)
        if task['payload'] is None:
            return
        await self.broadcast_to_all_fronts(task)

    # TODO: FINISH
    async def plot_xy_scatter(self, task):
        #  print(f'[Server] plotting data (xy scatter)!')
        plot_task, data_tasks = datastore.add_xy_data(task['payload'], self.process_id)
        for data_task in data_tasks:
            await self.broadcast_to_all_fronts(data_task)
        await self.broadcast_to_all_fronts(plot_task)


def get_ws_route_handler(mode):
    async def _router(request):
        ws_manager = SocketManager(mode)
        await ws_manager.prepare(request)
        await ws_manager.send_json(
            dict(
                type='WS_CONNECTION_ID',
                payload={'connection_id': ws_manager.connection_id},
                meta={},
            )
        )
        # Perform initialization functionality after sending connection ID
        await getattr(ws_manager, f'{mode}_initialize_connection')()
        async for msg in ws_manager:
            if msg.type == aiohttp.WSMsgType.TEXT:
                json_data = json.loads(msg.data)
                if isinstance(json_data, str):
                    json_data = json.loads(json_data)
                if json_data['type'] == 'disconnect':
                    print(f'Received disconnect for {ws_manager.connection_id}')
                    await ws_manager.close()
                else:
                    await ws_manager.handle_request(json_data, mode)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print('WS connection closed with exception %s' % ws_manager.exception())
        return ws_manager

    return _router


class SocketManager(web.WebSocketResponse, EngineMixin, FrontMixin):
    front_socket = get_ws_route_handler('front')
    engine_socket = get_ws_route_handler('engine')
    ws_count = 0
    ws_dict = {'engine': {}, 'front': {}}
    # Stores the connection ids for each process_id (engines)
    engine_connection_ids = {}

    @staticmethod
    def get_engine_ws(process_id):
        connection_id = SocketManager.engine_connection_ids[process_id]
        return SocketManager.ws_dict['engine'][connection_id]

    async def broadcast_to_all_fronts(self, task):
        await asyncio.gather(
            *[
                asyncio.create_task(x.send_json(task))
                for x in SocketManager.ws_dict['front'].values()
            ]
        )

    async def _send(self, type, payload=None, meta={'passive': True}):
        await self.send_json({'type': type, 'payload': None, 'meta': meta})

    def _print_socket_info(self):
        ws_dict = SocketManager.ws_dict
        print('> Socket Updates:')
        fronts = ', '.join(str(k) for k in ws_dict['front'].keys())
        backs = ', '.join(str(k) for k in ws_dict['engine'].keys())
        print(f'\tFront ends: {fronts}')
        print(f'\t   Engines: {backs}')
        print(f'\t    Latest: {self.connection_id}')

    def __init__(self, mode, **kwargs):
        assert mode in ['front', 'engine']
        super().__init__(**kwargs)
        self.mode = mode
        self.connection_id = SocketManager.ws_count
        SocketManager.ws_count += 1
        SocketManager.ws_dict[mode][self.connection_id] = self
        self._print_socket_info()

    async def close(self):
        if self.connection_id in SocketManager.ws_dict[self.mode]:
            del SocketManager.ws_dict[self.mode][self.connection_id]
        await getattr(self, f'{self.mode}_close')()
        await super().close()

    async def handle_request(self, data, mode):
        request = data["type"]
        #  print(f'[Server] Got {request} from {mode} ({self.connection_id})')
        try:
            handler = getattr(self, f'{request.lower()}')
        except AttributeError:
            handler = getattr(self, f'{mode}_default_handler')
        await handler(data)
