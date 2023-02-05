import { ConfusionMatrix } from '../confusion-matrix';
import { Metric } from '../metric';
import { MetricCalculator } from '../metric-calculator';
import { TestingResult } from '../result';

export class NPVMetricCalculator extends MetricCalculator {
  public calculate({ trueValues, predictedValues }: TestingResult): Metric {
    const { tn, fn } = new ConfusionMatrix(trueValues, predictedValues);

    return new Metric({
      title: 'NPV',
      value: tn / (tn + fn)
    });
  }
}
