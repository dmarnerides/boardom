import React from 'react';
import { connect } from 'react-redux';
import Plot from 'react-plotly.js';

// Actions
const actionNames = {
  PLOT_XY_SCATTER: 'PLOT_XY_SCATTER',
};

// Default state
const default_plots_state = {
  byId: {},
  allIds: [],
};

//Reducers
function plotsReducer(plots, action) {
  switch (action.type) {
    case actionNames.PLOT_XY_SCATTER: {
      // Payload is a dict with plotId, dataIds {xId, yId} and
      // Plot formatting parameters (and other things)
      // {{plotId, dataIds}}
      const id = action.payload.plotId;
      let newAllIds = plots.allIds;
      if (!newAllIds.includes(id)) {
        newAllIds = [...newAllIds, id];
      }

      const ret = {
        byId: {
          ...plots.byId,
          [id]: action.payload,
        },
        allIds: newAllIds,
      };
      return ret;
    }
    default:
      return plots;
  }
}

function AllPlotsComponentBase({ allPlotIds }) {
  return (
    <React.Fragment>
      <ul className="list-group">
        {allPlotIds.map(plotId => (
          <li key={plotId}>
            <PlotElementComponent plotId={plotId} />
          </li>
        ))}
      </ul>
    </React.Fragment>
  );
}

function PlotElementComponentBase(data) {
  return <Plot {...data} />;
}
//Mappers
const mapStateToProps = state => {
  return { allPlotIds: state.plots.allIds };
};
const mapStateToPropsElement = (state, ownProps) => {
  const plotId = ownProps.plotId;
  const plot = state.plots.byId[plotId];
  const dataIds = plot.dataIds;

  // data is array of objects [ {plot_1_stuff}, {plot_2_stuff}]
  // (Multiple graphs on the same plot
  const data = dataIds.reduce(
    (data_arr, subarr, subidx) => [
      ...data_arr,
      subarr.reduce(
        (obj, key) => ({
          ...obj,
          [plot.plotlyAxesById[key]]: state.data.byId[key].data,
        }),
        {
          ...plot.plotlyDataExtras[subidx],
        }
      ),
    ],
    []
  );

  const ret = {
    layout: plot.layout,
    data: data,
  };
  return ret;
};

// Connected Component
const AllPlotsComponent = connect(mapStateToProps)(AllPlotsComponentBase);
const PlotElementComponent = connect(mapStateToPropsElement)(
  PlotElementComponentBase
);
export { AllPlotsComponent, default_plots_state, plotsReducer };
