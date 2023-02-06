import { ConfusionMatrix } from '../confusion-matrix';
import { Metric } from '../metric';
import { MetricCalculator } from '../metric-calculator';
import { TestingResult } from '../result';

export class FNRMetricCalculator extends MetricCalculator {
  public calculate({ trueValues, predictedValues }: TestingResult): Metric {
    const { tp, fn } = new ConfusionMatrix(trueValues, predictedValues);

    return new Metric({
      title: 'FNR',
      value: fn / (fn + tp)
    });
  }
}
