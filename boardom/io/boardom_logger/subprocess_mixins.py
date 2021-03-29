import boardom as bd
from .lmdb_handler import LMDBHandler


class _SubrocessPrivateAPIMixin:
    def assign_meta_(self, task):
        meta = task.get('meta', {})
        meta.update(
            {
                'connection_id': self.connection_id,
                'process_id': self.process_id,
            }
        )
        task['meta'] = meta


class _ServerHandlerMixin:
    async def server_disconnect(self):
        print('[Daemon] Received disconnect from server. Closing WS.')
        await self.ws.close()
        return False

    async def server_ws_connection_id(self, task):
        self.connection_id = task['payload']['connection_id']
        print(f'[Daemon] Got assigned connection id: {self.connection_id}')
        return True

    async def server_default_handler(self, task):
        tt = task['type']
        payload = task.get('payload', None)
        print(f'[Daemon] Task from server: {tt}, Payload: {payload}')
        await self.to_parent.send_msgpack(task)
        return True


class _ParentHandlerMixin:
    async def parent_default_handler(self, task):
        tt = task['type']
        print(f'[Daemon] Task from parent: {tt}')
        await self.queue_for_ws_tasks.put(task)

    async def parent_start_lmdb(self, task):
        print('[Daemon] Starting LMDB')
        self.lmdb_handler = LMDBHandler(task['payload']['session_path'])

    async def parent_exit(self, task):
        print('[Daemon] Got parent EXIT task. Disconnecting from server.')
        self.should_exit = True
        await self.queue_for_ws_tasks.put(
            dict(payload=None, type='disconnect', meta={})
        )
        await self._close_ws()

    async def parent_plot_xy_scatter(self, task):
        # Generate id for x, y,
        payload = task['payload']
        name = payload['name']
        x_name, y_name = f'{name}_x', f'{name}_y'
        payload['x_name'] = x_name
        payload['y_name'] = y_name
        payload.update(
            dict(
                plot_id=str(hash((self.process_id, name))),
                x_id=str(hash((self.process_id, x_name))),
                y_id=str(hash((self.process_id, y_name))),
            )
        )

        await self.queue_for_ws_tasks.put(task)
        self.lmdb_handler.write_scalar(payload['x'], x_name, self.process_id)
        self.lmdb_handler.write_scalar(payload['y'], y_name, self.process_id)
