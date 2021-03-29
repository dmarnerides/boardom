import React from 'react';
import { connect } from 'react-redux';
import { wsSendAction } from './websocket';
import { default_group } from '../constants';
import {
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@material-ui/core';
import { makeStyles } from '@material-ui/core/styles';

// Actions
const actionNames = {
  REQUEST_CONFIG_STORE: 'REQUEST_CFG_STORE',
  // These actions are raised by the server, via the websocket
  CONFIG_RECEIVED_NEW_STORE: 'ENGINE_CFG_FULL',
  CONFIG_RECEIVED_NEW_VALUE: 'SET_CFG_VALUE',
};

function configRequestConfigStoreAction() {
  return wsSendAction({ type: actionNames.REQUEST_CONFIG_STORE });
}

// Default state
const default_configs_state = {
  byId: {},
  allIds: [],
};

// Reducers
function configsReducer(configs, action) {
  switch (action.type) {
    case actionNames.CONFIG_RECEIVED_NEW_STORE:
      // Payload is a dict with ids
      // { arg_id: {arg_name, group, value, tags, meta, process_id, arg_id}}}
      return {
        byId: {
          ...configs.byId,
          ...action.payload,
        },
        allIds: [
          ...new Set([...configs.allIds, ...Object.keys(action.payload)]),
        ],
      };
    case actionNames.CONFIG_RECEIVED_NEW_VALUE:
      // Payload is:
      // { value, arg_name, process_id, arg_id}
      return {
        byId: {
          ...configs.byId,
          [action.payload.id]: action.payload,
        },
        allIds: [...new Set([...configs.allIds, action.payload.id])],
      };
    default:
      return configs;
  }
}
const useStyles = makeStyles({
  table: {
    minWidth: 650,
  },
});

// Components
function Configs({ cfg_allIds, requestConfigList }) {
  const classes = useStyles();
  return (
    <React.Fragment>
      <Button variant="contained" color="primary" onClick={requestConfigList}>
        {' '}
        Refresh
      </Button>
      <TableContainer component={Paper}>
        <Table className={classes.table}>
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell>Value</TableCell>
              <TableCell>Group</TableCell>
              <TableCell>Process</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {cfg_allIds.map(cfg_id => (
              <ConfigElementComponent cfg_id={cfg_id} key={cfg_id} />
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </React.Fragment>
  );
}

function ConfigElement({ name, value, group, process_id }) {
  return (
    <TableRow>
      <TableCell>{name}</TableCell>
      <TableCell>{String(value)}</TableCell>
      <TableCell>{group}</TableCell>
      <TableCell>{process_id}</TableCell>
    </TableRow>
  );
}

// Mappers
const mapStateToPropsElement = (state, ownProps) => {
  const { name, value, group, process_id } = state.configs.byId[
    ownProps.cfg_id
  ];
  return { name, value, group, process_id };
};

const mapStateToProps = state => {
  return { cfg_allIds: state.configs.allIds };
};

const mapDispatchToProps = dispatch => {
  return {
    requestConfigList: () => {
      dispatch(configRequestConfigStoreAction());
    },
  };
};

// Connected Component
const ConfigElementComponent = connect(mapStateToPropsElement)(ConfigElement);
const ConfigsComponent = connect(
  mapStateToProps,
  mapDispatchToProps
)(Configs);

export { ConfigsComponent, configsReducer, default_configs_state, actionNames };
