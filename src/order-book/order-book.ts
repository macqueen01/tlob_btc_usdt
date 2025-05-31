export interface JSONable {
    toJSON(): JSON;
};

export interface CSVable {
    toRow(depth?: 10 | 20 | 50 | 80 | 120): (string | number)[];
}

export abstract class OrderBookChange {
    constructor(
        private readonly type: "SET" | "SNAPSHOT", // | "ADD" | "SUB" | 
        private readonly at: Date,
        private readonly price: number,
        private readonly size: number
    ) {};

    abstract get On(): string;
    
    get Type(): "SET" | "SNAPSHOT" {
        return this.type;
    };

    get At(): Date {
        return this.at;
    };

    get Price(): number {
        return this.price;
    };

    get Size(): number {
        return this.size;
    };
};

export interface OrderBook extends JSONable, CSVable {
    updateTo(dest: TimeSeries): void;
    apply(change: OrderBookChange): void;
};

export interface TimeSeries extends CSVable {
    Put(obj: JSONable, at: Date): void;
};