export class Metric {
  public title: string;
  public value: number;

  constructor(metric: Metric) {
    this.title = metric.title;
    this.value = metric.value;
  }
}
