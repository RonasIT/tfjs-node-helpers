import { ConfusionMatrix } from '../confusion-matrix';
import { Metric } from '../metric';
import { MetricCalculator } from '../metric-calculator';
import { TestingResult } from '../result';
import { AccuracyMetricCalculator } from './accuracy';

export class CohenKappaMetricCalculator extends MetricCalculator {
  public calculate({ trueValues, predictedValues, probabilities }: TestingResult): Metric {
    const { tp, tn, fp, fn } = new ConfusionMatrix(trueValues, predictedValues);
    const accuracy = (new AccuracyMetricCalculator()).calculate({ trueValues, predictedValues, probabilities }).value;
    const numberOfSamples = tp + tn + fp + fn;
    const expectedAccuracy = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / numberOfSamples ** 2;

    return new Metric({
      title: 'Cohen Kappa',
      value: (accuracy - expectedAccuracy) / (1 - expectedAccuracy)
    });
  }
}
