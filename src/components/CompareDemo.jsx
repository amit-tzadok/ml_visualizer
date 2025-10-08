import React from 'react';
import PerceptronDemo from './PerceptronDemo';

const CompareDemo = ({ leftType = 'linear', rightType = 'poly' }) => {
  return (
    <div style={{ display: 'flex', gap: 16 }}>
      <div>
        <h3 style={{ textAlign: 'center' }}>{leftType}</h3>
        <PerceptronDemo classifierType={leftType} />
      </div>
      <div>
        <h3 style={{ textAlign: 'center' }}>{rightType}</h3>
        <PerceptronDemo classifierType={rightType} />
      </div>
    </div>
  );
};

export default CompareDemo;
