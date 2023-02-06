import { math, Tensor } from '@tensorflow/tfjs-node';

export class ConfusionMatrix {
  public tp: number;
  public tn: number;
  public fp: number;
  public fn: number;

  constructor(trueValues: Tensor, predictedValues: Tensor) {
    const trueY = trueValues.as1D();
    const predY = predictedValues.as1D();
    const numberOfClasses = 2;

    const confusionMatrix = math.confusionMatrix(trueY, predY, numberOfClasses);

    trueY.dispose();
    predY.dispose();

    const [[tn, fp], [fn, tp]] = confusionMatrix.arraySync();

    confusionMatrix.dispose();

    this.tp = tp;
    this.fp = fp;
    this.fn = fn;
    this.tn = tn;
  }
}
