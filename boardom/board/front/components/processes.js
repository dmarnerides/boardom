import React from 'react';
import { connect } from 'react-redux';
import { wsSendAction } from './websocket';

//Actions
const actionNames = {
  // Internal
  PROCESS_TOGGLED: 'PROCESS_TOGGLED',
  // From server
  UPDATE_PROCESS_INFO: 'UPDATE_PROCESS_INFO',
  NEW_PROCESS_LIST_ACQUIRED: 'NEW_PROCESS_LIST_ACQUIRED',
  PROCESS_DEACTIVATED: 'PROCESS_DEACTIVATED',
  // From server (interceptions from other actions)
  SESSION_PATH_ACQUIRED: 'ENGINE_SESSION_PATH',
  // To server
  PROCESS_LIST_REQUESTED: 'PROCESS_LIST_REQUESTED',
};

function processListRequestAction() {
  return wsSendAction({ type: actionNames.PROCESS_LIST_REQUESTED });
}

function processToggleAction(process_id) {
  return {
    type: actionNames.PROCESS_TOGGLED,
    payload: process_id,
  };
}

// Default state
// Each entry in 'byId' is like this:
// [id]: {
//   id: id
//   path: str or null
//   active: bool
//   toggled: bool # For checkboxes
// }
const default_processes_state = {
  byId: {},
  allIds: [],
};

function processesReducer(processes, action) {
  switch (action.type) {
    case actionNames.PROCESS_TOGGLED: {
      // payload: process_id
      const id = action.payload;
      return {
        ...processes,
        byId: {
          ...processes.byId,
          [id]: { ...processes.byId[id], toggled: !processes.byId[id].toggled },
        },
      };
    }
    case actionNames.PROCESS_DEACTIVATED: {
      // payload: process_id
      const id = action.payload;
      return {
        ...processes,
        byId: {
          ...processes.byId,
          [id]: { ...processes.byId[id], active: false },
        },
      };
    }
    case actionNames.NEW_PROCESS_LIST_ACQUIRED: {
      // payload: [{}, ...]
      const new_processes_by_id = action.payload.reduce((obj, processObj) => {
        const process = processObj.id;
        return {
          ...obj,
          [process]: {
            ...processObj,
            toggled: processes.byId[process]
              ? processes.byId[process].toggled
              : false,
          },
        };
      }, {});
      return {
        ...processes,
        byId: {
          ...processes.byId,
          ...new_processes_by_id,
        },
        allIds: [
          ...new Set([...processes.allIds, ...action.payload.map(a => a.id)]),
        ],
      };
    }
    case actionNames.UPDATE_PROCESS_INFO: {
      // payload: process_id
      const process = action.payload.id;
      return {
        ...processes,
        byId: {
          ...processes.byId,
          [process]: {
            ...action.payload,
            toggled: processes.byId[process]
              ? processes.byId[process].toggled
              : false,
          },
        },
        allIds: [...new Set([...processes.allIds, process])],
      };
    }
    case actionNames.SESSION_PATH_ACQUIRED: {
      //payload: session_path, meta: connection_id, process_id
      const pid = action.meta.process_id;
      if (pid in processes.byId) {
        return {
          ...processes,
          byId: {
            ...processes.byId,
            [pid]: {
              ...processes.byId[pid],
              path: action.payload,
            },
          },
        };
      }
      return processes;
    }
    default:
      return processes;
  }
}

//Components
class Processes extends React.Component {
  render() {
    const process_ids = this.props.process_ids;
    return (
      <React.Fragment>
        <ul className="list-group">
          {process_ids.map(process_id => (
            <li key={process_id}>
              <ProcessElementComponent id={process_id} />
            </li>
          ))}
        </ul>
      </React.Fragment>
    );
  }
}

function ProcessElement({ id, toggled, path, active, toggleElement }) {
  let disptext = path === null ? id : `${path} (${id})`;
  disptext += active ? ' - active' : '';
  return (
    <div className="form-check">
      <input
        className="form-check-input"
        type="checkbox"
        value=""
        id={id}
        checked={toggled}
        onChange={() => toggleElement(id)}
      />
      <label className="form-check-label" htmlFor={id}>
        {disptext}
      </label>
    </div>
  );
}

// Mappers
const mapStateToProps = state => {
  return {
    process_ids: state.processes.allIds,
  };
};

const mapStateToPropsElement = (state, ownProps) => {
  return {
    toggled: state.processes.byId[ownProps.id].toggled,
    path: state.processes.byId[ownProps.id].path,
    active: state.processes.byId[ownProps.id].active,
  };
};

const mapDispatchToPropsElement = dispatch => {
  return {
    toggleElement: id => {
      dispatch(processToggleAction(id));
    },
  };
};

// Connected Component
const ProcessesComponent = connect(mapStateToProps)(Processes);
const ProcessElementComponent = connect(
  mapStateToPropsElement,
  mapDispatchToPropsElement
)(ProcessElement);

export { ProcessesComponent, processesReducer, default_processes_state };
