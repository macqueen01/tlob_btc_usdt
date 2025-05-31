import { OrderBook, OrderBookChange, TimeSeries } from "../order-book";

export class SellOrderBookChange extends OrderBookChange {
    get On(): "SELL" {
        return "SELL"
    };
};

export class SellOrderBook implements OrderBook {
    private priceToSize: Record<number, number> = {};
    private snapshots: Record<number, number> = {};
    private recentChange: Date = new Date;
    private sortedPrices: number[] = [];  // Keep track of sorted prices

    apply(change: SellOrderBookChange): void {
        if (change.On != "SELL") throw new Error("Applying Wrong Change to SellOrderBook");

        this.recentChange = change.At;

        if (change.Type == "SNAPSHOT") {
            this.snapshots[change.Price] = change.Size;
            return;
        };

        if (!this.isSnapshotEmpty()) {
            this.loadSnapshots();
        };

        if (change.Type == "SET") {
            this.priceToSize[change.Price] = change.Size;
            
            // Update sortedPrices
            const index = this.sortedPrices.indexOf(change.Price);
            if (index === -1 && change.Size !== 0) {
                // Insert new price in sorted order
                this.sortedPrices.push(change.Price);
                this.sortedPrices.sort((a, b) => a - b);  // Ascending for sell
            } else if (change.Size === 0) {
                // Remove price if size is 0
                this.sortedPrices.splice(index, 1);
            }
            return;
        };
    };

    truncate(depth: 10 | 20 | 50 | 80 | 120 = 10): void {
        if (this.sortedPrices.length <= depth) {
            return;
        }
        
        const pricesToKeep = this.sortedPrices.slice(0, depth);
        const newPriceToSize: Record<number, number> = {};
        
        pricesToKeep.forEach(price => {
            newPriceToSize[price] = this.priceToSize[price];
        });
        
        this.priceToSize = newPriceToSize;
        this.sortedPrices = pricesToKeep;
    }

    loadSnapshots(): void {
        this.drain();
        Object.entries(this.snapshots).forEach((entry) => {
            const price = Number(entry[0]);
            const size = entry[1];
            this.priceToSize[price] = size;
            this.sortedPrices.push(price);
        });
        this.sortedPrices.sort((a, b) => a - b);  // Ascending for sell
        this.snapshots = {};
    };

    private drain(): void { 
        this.priceToSize = {};
        this.sortedPrices = [];
    };

    private isSnapshotEmpty(): boolean {
        return Object.keys(this.snapshots).length === 0;
    }

    updateTo(dest: TimeSeries): void {
        dest.Put(this, this.recentChange);
    };

    toJSON(): JSON {
        return JSON.parse(JSON.stringify(
            this.sortedPrices.map(price => ({
                price,
                size: this.priceToSize[price]
            }))
        ));
    }

    toRow(depth: 10 | 20 | 50 | 80 | 120 = 10): (string | number)[] {
        return this.sortedPrices
            .slice(0, depth)
            .flatMap(price => [price, this.priceToSize[price]]);
    }
}