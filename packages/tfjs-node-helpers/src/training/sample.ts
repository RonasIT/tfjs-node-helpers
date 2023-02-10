import { TensorContainerObject } from "@tensorflow/tfjs-node";

export type Sample = {
  input: TensorContainerObject;
  output: TensorContainerObject;
};
