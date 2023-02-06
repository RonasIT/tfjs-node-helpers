import { metrics } from '@tensorflow/tfjs-node';
import { Metric } from '../metric';
import { MetricCalculator } from '../metric-calculator';
import { TestingResult } from '../result';

export class BrierLossMetricCalculator extends MetricCalculator {
  public calculate({ trueValues, probabilities }: TestingResult): Metric {
    const trueY = trueValues.as1D();
    const predY = probabilities.as1D();
    const valueTensor = metrics.MSE(trueY, predY);

    trueY.dispose();
    predY.dispose();

    const [value] = valueTensor.dataSync();

    valueTensor.dispose();

    return new Metric({
      title: 'Brier Loss',
      value
    });
  }
}
