import { binarize } from '../../utils/binarize';
import { Metric } from '../metric';
import { MetricCalculator } from '../metric-calculator';
import { TestingResult } from '../result';
import { FPRMetricCalculator } from './fpr';
import { RecallMetricCalculator } from './recall';

export class ROCAUCMetricCalculator extends MetricCalculator {
  constructor(private numberOfSteps: number = 20) {
    super();
  }

  public calculate({ trueValues, probabilities }: TestingResult): Metric {
    const numberOfThresholds = this.numberOfSteps + 1;
    const thresholdStepSize = 1 / this.numberOfSteps;
    const tprs: Array<number> = [];
    const fprs: Array<number> = [];
    const fprCalculator = new FPRMetricCalculator();
    const tprCalculator = new RecallMetricCalculator();
    let area = 0;
    let threshold = 0;

    for (let i = 0; i < numberOfThresholds; i++) {
      const predictedValues = binarize(probabilities, threshold);

      fprs.push(fprCalculator.calculate({ trueValues, probabilities, predictedValues }).value);
      tprs.push(tprCalculator.calculate({ trueValues, probabilities, predictedValues }).value);

      predictedValues.dispose();

      if (i > 0) {
        area += (tprs[i] + tprs[i - 1]) * (fprs[i - 1] - fprs[i]) / 2;
      }

      threshold += thresholdStepSize;
    }

    return new Metric({
      title: 'ROC AUC',
      value: area
    });
  }
}
