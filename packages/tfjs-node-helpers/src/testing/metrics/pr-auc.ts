import { binarize } from '../../utils/binarize';
import { Metric } from '../metric';
import { MetricCalculator } from '../metric-calculator';
import { TestingResult } from '../result';
import { PrecisionMetricCalculator } from './precision';
import { RecallMetricCalculator } from './recall';

export class PRAUCMetricCalculator extends MetricCalculator {
  constructor(private numberOfSteps: number = 20) {
    super();
  }

  public calculate({ trueValues, probabilities }: TestingResult): Metric {
    const numberOfThresholds = this.numberOfSteps + 1;
    const thresholdStepSize = 1 / this.numberOfSteps;
    const precisions: Array<number> = [];
    const recalls: Array<number> = [];
    const precisionCalculator = new PrecisionMetricCalculator();
    const recallCalculator = new RecallMetricCalculator();
    let area = 0;
    let threshold = 0;

    for (let i = 0; i < numberOfThresholds; i++) {
      const predictedValues = binarize(probabilities, threshold);

      precisions.push(precisionCalculator.calculate({ trueValues, probabilities, predictedValues }).value);
      recalls.push(recallCalculator.calculate({ trueValues, probabilities, predictedValues }).value);

      predictedValues.dispose();

      if (i > 0) {
        area += (precisions[i] + precisions[i - 1]) * (recalls[i - 1] - recalls[i]) / 2;
      }

      threshold += thresholdStepSize;
    }

    return new Metric({
      title: 'PR AUC',
      value: area
    });
  }
}
