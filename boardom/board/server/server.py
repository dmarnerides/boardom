import os
import asyncio
from aiohttp import web
from .datastore import datastore
from .socket_manager import SocketManager
from .common import DIST_PATH, INDEX_PATH, STATICS, ALL_FILES


async def index_route(request):
    if request.path in STATICS:
        return web.FileResponse(os.path.join(DIST_PATH, request.path))
    return web.FileResponse(INDEX_PATH)


def run_server():
    web.run_app(create_server(), host='127.0.0.1', port=8089)


def create_server():
    async def reroute_middleware(app, _router):
        async def middleware_router(request):
            # All requests are renormalized to /{something}
            # {something} must be in ALL_FILES, otherwise redirect
            new_path = [x for x in request.path.split('/') if x]
            new_path = '' if not new_path else new_path[-1]
            if (
                new_path == '' or new_path in ALL_FILES
            ) and '/' + new_path == request.path:
                request = request.clone(rel_url=new_path)
            else:
                raise web.HTTPFound('/')
            try:
                response = await _router(request)
                if response.status == 404:
                    response = await index_route(request)
                return response
            except web.HTTPException as ex:
                if ex.status == 404:
                    return await index_route(request)
                raise

        return middleware_router

    async def on_startup(app):
        print('Boardom launching')
        app['bd.data'] = datastore.create()
        app['async_datastore_init'] = asyncio.create_task(datastore.initialize())
        print('Done!')

    async def on_cleanup(app):
        app['async_datastore_init'].cancel()
        await app['async_datastore_init']

    app = web.Application(middlewares=[reroute_middleware])
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.add_routes(
        [
            web.get('/', index_route),
            web.get('/front_socket', SocketManager.front_socket),
            web.get('/engine_socket', SocketManager.engine_socket),
        ]
    )
    return app
