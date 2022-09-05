export class Feature<T> {
  public type: T;
  public label: string;
  public value: number;

  constructor(feature: Feature<T>) {
    this.type = feature.type;
    this.label = feature.label;
    this.value = feature.value;
  }
}
