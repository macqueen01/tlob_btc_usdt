import { GoWrapper } from './go-integration/go-wrapper';

// Example of how to integrate Go processing into your existing order book system
export async function exampleGoIntegration() {
    const goWrapper = new GoWrapper();

    // Sample order book data (matching your existing interfaces)
    const sampleOrderBook = {
        bids: [
            {
                type: "SET",
                at: new Date("2025-01-05T10:00:00Z"),
                price: 50000.50,
                size: 1.5,
                side: "bid"
            },
            {
                type: "SET", 
                at: new Date("2025-01-05T10:00:01Z"),
                price: 50000.25,
                size: 2.0,
                side: "bid"
            }
        ],
        asks: [
            {
                type: "SET",
                at: new Date("2025-01-05T10:00:00Z"),
                price: 50001.00,
                size: 1.0,
                side: "ask"
            },
            {
                type: "SET",
                at: new Date("2025-01-05T10:00:01Z"),
                price: 50001.25,
                size: 0.8,
                side: "ask"
            }
        ]
    };

    try {
        console.log("Processing order book with Go...");
        const result = await goWrapper.processOrderBookWithGo(sampleOrderBook);
        console.log("Go processing result:");
        console.log(result);

        // You can also build the Go program once and run the binary multiple times for better performance
        console.log("Building Go binary...");
        await goWrapper.buildGoProgram('cmd/processor/main.go', 'bin/processor');
        
        console.log("Running Go binary...");
        const binaryResult = await goWrapper.runGoBinary('./bin/processor', ['temp_orderbook.json']);
        console.log("Binary result:", binaryResult);

    } catch (error) {
        console.error("Error processing with Go:", error);
    }
}

// You can call this from your main function
if (require.main === module) {
    exampleGoIntegration();
} 