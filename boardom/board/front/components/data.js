import React from 'react';
import { connect } from 'react-redux';

// Actions
const actionNames = {
  // These actions are raised by the server, via the websocket
  RECEIVED_NEW_DATA: 'RECEIVED_NEW_DATA',
};
const dataTypeNames = {
  SCALAR: 'SCALAR',
  SCALAR_LIST: 'SCALAR_LIST',
};

// Default state
const default_data_state = {
  byId: {},
  allIds: [],
};

// Reducers
function dataSubReducer(currentData, dataPoint, dataType) {
  switch (dataType) {
    case dataTypeNames.SCALAR:
      if (currentData === null) {
        return [dataPoint];
      }
      return [...currentData, dataPoint];
    case dataTypeNames.SCALAR_LIST:
      if (currentData === null) {
        return [...dataPoint];
      }
      return [...currentData, ...dataPoint];
    default:
      return currentData;
  }
}

// main reducer
function dataReducer(data, action) {
  switch (action.type) {
    case actionNames.RECEIVED_NEW_DATA: {
      // Payload is a dict with dataId and data:
      // {payload: {dataId, data}, meta: {dataType}}
      // datapoint is an object
      const id = action.payload.dataId;
      let current_data = null;
      let new_allIds = data.allIds;
      if (!data.allIds.includes(id)) {
        new_allIds = [...data.allIds, id];
      } else {
        current_data = data.byId[id].data;
      }
      const ret = {
        byId: {
          ...data.byId,
          [id]: {
            data: dataSubReducer(
              current_data,
              action.payload.datapoint,
              action.meta.dataType
            ),
            dataType: action.meta.dataType,
          },
        },
        allIds: new_allIds,
      };
      return ret;
    }
    default:
      return data;
  }
}

// Components
function DataList({ allDataIds }) {
  return (
    <React.Fragment>
      <table className="table">
        <thead>
          <tr className="thread-dark">
            <th scope="col">Id</th>
            <th scope="col">DataType</th>
          </tr>
        </thead>
        <tbody>
          {allDataIds.map(dataId => (
            <DataElementComponent dataId={dataId} key={dataId} />
          ))}
        </tbody>
      </table>
    </React.Fragment>
  );
}

function DataElement({ dataId, dataType }) {
  return (
    <tr>
      <td>{dataId}</td>
      <td>{dataType}</td>
    </tr>
  );
}

// Mappers
const mapStateToPropsElement = (state, ownProps) => {
  const dataId = ownProps.dataId;
  const dataType = state.data.byId[dataId].dataType;
  return { dataId, dataType };
};

const mapStateToProps = state => {
  return { allDataIds: state.data.allIds };
};

// Connected Component
const DataElementComponent = connect(mapStateToPropsElement)(DataElement);
const DataListComponent = connect(mapStateToProps)(DataList);

export { DataListComponent, dataReducer, default_data_state, actionNames };
