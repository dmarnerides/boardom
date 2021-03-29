import { createStore, applyMiddleware } from 'redux';
import { configsReducer, default_configs_state } from './config';
import { processesReducer, default_processes_state } from './processes';
import { dataReducer, default_data_state } from './data';
import { plotsReducer, default_plots_state } from './plot';
import {
  wsMiddleware,
  actionNames as wsActionNames,
  wsReducer,
  default_ws_state,
} from './websocket';
import thunk from 'redux-thunk';

const default_state = {
  configs: default_configs_state,
  ws: default_ws_state,
  processes: default_processes_state,
  data: default_data_state,
  plots: default_plots_state,
};

function delegateReducer(state, action) {
  return {
    configs: configsReducer(state.configs, action),
    ws: wsReducer(state.ws, action),
    processes: processesReducer(state.processes, action),
    data: dataReducer(state.data, action),
    plots: plotsReducer(state.plots, action),
  };
}

function mainReducer(state = default_state, action) {
  if (action.type === wsActionNames.WS_MESSAGE_RECEIVED) {
    // If action is not from the websocket, propagate to reducers
    action = action.payload;
  }
  return delegateReducer(state, action);
}

const Store = createStore(mainReducer, applyMiddleware(thunk, wsMiddleware()));

export { Store };
