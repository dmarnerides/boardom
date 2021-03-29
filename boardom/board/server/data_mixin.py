import uuid


# Future: Add functionality for querying the name of the automatically
# generated data names given a process_id and a plot id


class _DataMixin:
    # TODO: FINISH
    # MUST ADD INITIALIZATION AND LOADING FROM DISK

    # TODO: Handle process changes
    def handle_process_change(self, old_process_id, new_process_id):
        pass

    def add_xy_data(self, payload, process_id):
        plot_name, plot_id = payload['name'], payload['plot_id']
        x_id, y_id = payload['x_id'], payload['y_id']
        vis_task = {
            'type': 'PLOT_XY_SCATTER',
            'payload': {
                'plotId': plot_id,
                'dataIds': [[x_id, y_id]],
                'plotlyAxesById': {x_id: 'x', y_id: 'y'},
                'plotlyDataExtras': {
                    'type': 'scatter',
                    'mode': 'lines+points',
                    'marker': {'color': 'green'},
                },
                'layout': {'width': 640, 'height': 480, 'title': plot_name},
            },
            'meta': {},
        }
        vis_store = self.store['visualisations']
        if plot_id not in vis_store:
            vis_store[plot_id] = vis_task
        else:
            vis_task.update(vis_store[plot_id])

        data_tasks = [
            {
                'type': 'RECEIVED_NEW_DATA',
                'payload': dict(dataId=d_id, datapoint=d_data),
                'meta': dict(dataType='SCALAR'),
            }
            for d_id, d_data in [(x_id, payload['x']), (y_id, payload['y'])]
        ]

        return vis_task, data_tasks
