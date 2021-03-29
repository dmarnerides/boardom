import React from 'react';
import { AllPlotsComponent } from './plot';
import { DataListComponent } from './data';

function HomeComponent() {
  return (
    <React.Fragment>
      <DataListComponent />
      <AllPlotsComponent />
    </React.Fragment>
  );
}

export { HomeComponent };
