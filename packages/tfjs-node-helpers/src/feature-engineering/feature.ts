export type FeatureOptions<T> = {
  type: T;
  label: string;
  value: number;
};

export class Feature<T> {
  public id: string;
  public type: T;
  public label: string;
  public value: number;

  constructor(feature: FeatureOptions<T>) {
    this.type = feature.type;
    this.label = feature.label;
    this.value = feature.value;
    this.id = String(this.type);
  }
}
