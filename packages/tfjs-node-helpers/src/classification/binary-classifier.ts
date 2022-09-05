import { LayersModel, loadLayersModel, tensor, Tensor } from '@tensorflow/tfjs-node';

export type BinaryClassifierOptions = {
  model: LayersModel;
};

export class BinaryClassifier {
  protected model?: LayersModel;

  constructor(options?: BinaryClassifierOptions) {
    this.model = options?.model;
  }

  public async load(path: string): Promise<void> {
    this.model = await loadLayersModel(`file://${path}`);
  }

  public async predict(input: Array<number>): Promise<number> {
    if (this.model === undefined) {
      throw new Error('Unable to predict, because the model is not loaded!');
    }

    const inputTensor = tensor(input, [1, input.length]);
    const outputTensor = this.model.predict(inputTensor) as Tensor;

    inputTensor.dispose();

    const [prediction] = await outputTensor.data();

    outputTensor.dispose();

    return prediction;
  }
}
