import { Task } from '../core/task';

export type ComposeOptions = {
  tasks: Array<Task>;
  outputFeatures?: Array<Task>;
};

export function compose(options: ComposeOptions): Task {
  return new Task({
    inputFeatures: [],
    outputFeatures: []
  });
}
