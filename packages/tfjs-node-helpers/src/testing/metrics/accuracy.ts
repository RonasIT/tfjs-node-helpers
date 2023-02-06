import { metrics } from '@tensorflow/tfjs-node';
import { Metric } from '../metric';
import { MetricCalculator } from '../metric-calculator';
import { TestingResult } from '../result';

export class AccuracyMetricCalculator extends MetricCalculator {
  public calculate({ trueValues, predictedValues }: TestingResult): Metric {
    const trueY = trueValues.as1D();
    const predY = predictedValues.as1D();
    const valueTensor = metrics.binaryAccuracy(trueY, predY);

    trueY.dispose();
    predY.dispose();

    const [value] = valueTensor.dataSync();

    valueTensor.dispose();

    return new Metric({
      title: 'Accuracy',
      value
    });
  }
}
