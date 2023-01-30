import { Metric } from './metric';
import { MetricCalculator } from './metric-calculator';

export const calculateMetrics = ({
  trueValues,
  predictedValues,
  metricCalculators
}: {
  trueValues: Float32Array;
  predictedValues: Float32Array;
  metricCalculators: Array<MetricCalculator>;
}): Array<Metric> => metricCalculators.map((calculator) => calculator.calculate(trueValues, predictedValues));
