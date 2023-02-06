import { Tensor } from "@tensorflow/tfjs-node";

export type TestingResult = {
  trueValues: Tensor;
  probabilities: Tensor;
  predictedValues: Tensor;
};
