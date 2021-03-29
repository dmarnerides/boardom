import React from 'react';
import { constants } from '../constants';
import { connect } from 'react-redux';
// Actions
const actionNames = {
  WS_CONNECT: 'WS_CONNECT', // Handled by middleware
  WS_CONNECTED: 'WS_CONNECTED', // Handled by reducer
  WS_DISCONNECT: 'WS_DISCONNECT', // Handled by middleware
  WS_DISCONNECTED: 'WS_DISCONNECTED', // Handled by reducer
  WS_MESSAGE_RECEIVED: 'WS_MESSAGE_RECEIVED', // Handled by mainReducer
  WS_SEND: 'WS_SEND', // Handled by middleware
  WS_CONNECTION_ID: 'WS_CONNECTION_ID', // Handled by reducer
};

function wsConnectAction(address) {
  return {
    type: actionNames.WS_CONNECT,
    payload: { address },
  };
}
function wsConnectedAction(address) {
  return {
    type: actionNames.WS_CONNECTED,
    payload: { address },
  };
}
function wsDisconnectedAction() {
  return {
    type: actionNames.WS_DISCONNECTED,
  };
}
function wsReceiveAction(payload) {
  // The payload is the "action" created from the server
  return {
    type: actionNames.WS_MESSAGE_RECEIVED,
    payload,
  };
}

function wsSendAction(payload) {
  return {
    type: actionNames.WS_SEND,
    payload,
  };
}

// Default state
const default_ws_state = { connected: false, connection_id: null };

// Reducer
function wsReducer(state, action) {
  switch (action.type) {
    case actionNames.WS_CONNECTED:
      return Object.assign({}, state, { connected: true });
    case actionNames.WS_DISCONNECTED:
      return Object.assign({}, state, { connected: false });
    case actionNames.WS_CONNECTION_ID:
      return Object.assign({}, state, { connection_id: action.payload.id });
    default:
      return state;
  }
}

// Middleware. Based on:
// https://dev.to/aduranil/how-to-use-websockets-with-redux-a-step-by-step-guide-to-writing-understanding-connecting-socket-middleware-to-your-project-km3
const wsMiddleware = () => {
  let ws = null;
  const onOpen = store => event => {
    store.dispatch(wsConnectedAction(event.target.url));
  };

  const onClose = store => () => {
    store.dispatch(wsDisconnectedAction());
  };

  const onMessage = store => event => {
    const action = JSON.parse(event.data);
    store.dispatch(wsReceiveAction(action));
  };

  let unsentTasks = [];

  return store => next => action => {
    switch (action.type) {
      case actionNames.WS_CONNECT:
        if (ws !== null) {
          ws.close();
        }
        ws = new WebSocket(action.payload.address);
        ws.onmessage = onMessage(store);
        ws.onclose = onClose(store);
        ws.onopen = onOpen(store);
        return next(action);
      case actionNames.WS_CONNECTED:
        // send unsent tasks
        // Maybe add this to its own middleware that handles unsent tasks
        // to the websocket (in case more accumulate after connection? )
        unsentTasks.forEach(task => store.dispatch(wsSendAction(task)));
        unsentTasks = [];
        return next(action);
      case actionNames.WS_DISCONNECT:
        if (ws !== null) {
          ws.close();
        }
        ws = null;
        return next(action);
      case actionNames.WS_SEND: {
        const task = JSON.stringify(action.payload);
        if (ws !== null && ws.readyState === 1) {
          ws.send(task);
        } else {
          unsentTasks.push(task);
        }
        return next(action);
      }
      default:
        return next(action);
    }
  };
};

class WsComponentBase extends React.Component {
  componentDidMount() {
    this.props.dispatch(wsConnectAction(constants.server_ws_address));
  }
  render() {
    return null;
  }
}

const WsComponent = connect()(WsComponentBase);

export {
  wsConnectAction,
  wsMiddleware,
  actionNames,
  wsReducer,
  default_ws_state,
  wsSendAction,
  WsComponent,
};
