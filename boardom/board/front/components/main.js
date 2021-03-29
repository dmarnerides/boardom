import React, { Fragment } from 'react';
import { connect } from 'react-redux';
import { Tabs, Tab } from '@material-ui/core';
import { Route, Link, Switch, Redirect, BrowserRouter } from 'react-router-dom';
import { ConfigsComponent } from './config';
import { ProcessesComponent } from './processes';
import { HomeComponent } from './home';
import { SettingsComponent } from './settings';
import { WsComponent } from './websocket';
import { AllPlotsComponent } from './plot';

function ViewComponent() {
  return (
    <BrowserRouter>
      <Fragment>
        <Route
          path="/"
          render={({ location }) => (
            <Fragment>
              <Tabs value={location.pathname}>
                <Tab label="Home" value="/" component={Link} to="/" />
                <Tab
                  label="Configs"
                  value="/configs"
                  component={Link}
                  to="/configs"
                />
                <Tab
                  label="Settings"
                  value="/settings"
                  component={Link}
                  to="/settings"
                />
                <Tab
                  label="Plots"
                  value="/plots"
                  component={Link}
                  to="/plots"
                />
              </Tabs>
              <ProcessesComponent />
              <Switch>
                <Route path="/configs" component={ConfigsComponent} />
                <Route path="/plots" component={AllPlotsComponent} />
                <Route
                  path="/settings"
                  render={() => <Fragment>Settings</Fragment>}
                />
                <Route path="/" render={() => <Fragment>Boardom</Fragment>} />
              </Switch>
            </Fragment>
          )}
        />
      </Fragment>
    </BrowserRouter>
  );
}

class Main extends React.Component {
  render() {
    return (
      <React.Fragment>
        <WsComponent />
        <ViewComponent />
      </React.Fragment>
    );
  }
}

const MainComponent = connect()(Main);

export { MainComponent };
