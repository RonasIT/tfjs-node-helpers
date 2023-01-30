export class ConfusionMatrix {
  public tp: number;
  public tn: number;
  public fp: number;
  public fn: number;

  constructor(trueValues: Float32Array, predictedValues: Float32Array) {
    this.tp = 0;
    this.tn = 0;
    this.fp = 0;
    this.fn = 0;

    for (let index = 0; index < trueValues.length; index++) {
      const trueValue = trueValues[index];
      const predictedValue = predictedValues[index];

      if (trueValue === 1 && predictedValue === 1) {
        this.tp++;
      } else if (trueValue === 0 && predictedValue === 0) {
        this.tn++;
      } else if (trueValue === 0 && predictedValue === 1) {
        this.fp++;
      } else {
        this.fn++;
      }
    }
  }
}
