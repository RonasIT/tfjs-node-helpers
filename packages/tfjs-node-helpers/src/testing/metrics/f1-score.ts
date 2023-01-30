import { ConfusionMatrix } from '../confusion-matrix';
import { Metric } from '../metric';
import { MetricCalculator } from '../metric-calculator';

export class F1ScoreMetricCalculator extends MetricCalculator {
  public calculate(trueValues: Float32Array, predictedValues: Float32Array): Metric {
    const { tp, fp, fn } = new ConfusionMatrix(trueValues, predictedValues);
    const precision = tp / (tp + fp);
    const recall = tp / (tp + fn);

    return new Metric({
      title: 'F1Score',
      value: (2 * precision * recall) / (precision + recall)
    });
  }
}
