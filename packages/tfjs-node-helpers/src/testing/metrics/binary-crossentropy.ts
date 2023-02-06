import { metrics } from '@tensorflow/tfjs-node';
import { Metric } from '../metric';
import { MetricCalculator } from '../metric-calculator';
import { TestingResult } from '../result';

export class BinaryCrossentropyMetricCalculator extends MetricCalculator {
  public calculate({ trueValues, probabilities }: TestingResult): Metric {
    const trueY = trueValues.as1D();
    const predY = probabilities.as1D();
    const valueTensor = metrics.binaryCrossentropy(trueY, predY);

    trueY.dispose();
    predY.dispose();

    const [value] = valueTensor.dataSync();

    valueTensor.dispose();

    return new Metric({
      title: 'Binary Cross-entropy',
      value
    });
  }
}
