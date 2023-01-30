import { ConfusionMatrix } from '../confusion-matrix';
import { Metric } from '../metric';
import { MetricCalculator } from '../metric-calculator';

export class AccuracyMetricCalculator extends MetricCalculator {
  public calculate(trueValues: Float32Array, predictedValues: Float32Array): Metric {
    const { tp, tn, fp, fn } = new ConfusionMatrix(trueValues, predictedValues);

    return new Metric({
      title: 'Accuracy',
      value: (tp + tn) / (tp + tn + fp + fn)
    });
  }
}
