import React from 'react'; // eslint-disable-line
import ReactDOM from 'react-dom';
import { Provider } from 'react-redux';
// import 'bootstrap/dist/css/bootstrap.min.css';
// This import requires static-loader for webpack
import './static/css/app.css';
import { MainComponent, Store } from './components';

ReactDOM.render(
  <Provider store={Store}>
    <MainComponent />
  </Provider>,
  document.getElementById('root')
);
