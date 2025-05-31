import { S3_CONFIG } from "../config";

export const S3_REGION = S3_CONFIG.region;
export const BUCKET = S3_CONFIG.bucket;
export const ACCESS_KEY = S3_CONFIG.accessKey;
export const SECRET_KEY = S3_CONFIG.secretKey;
export const S3_ENDPOINT = S3_CONFIG.endpoint;

export const TRAINING_LIST = [
    // Bullish 📈
    new Date("2023-03-14"), // CPI cooldown → BTC +10%
    new Date("2023-10-23"), // ETF speculation pump
    new Date("2024-02-13"), // CPI low → risk-on rally
    new Date("2025-01-10"), // Post-ETF approval rally
    
    // Bearish 📉
    new Date("2023-03-10"), // SVB collapse panic
    new Date("2023-11-09"), // Binance/DOJ fears, dump
    new Date("2024-08-17"), // Weekend low-liquidity drop
    new Date("2025-02-24"), // Sharp correction post-rally
    
    // Neutral 😐
    new Date("2023-06-17"), // Weekend, stable price
    new Date("2024-04-20"), // Bitcoin halving, muted intraday
    new Date("2024-12-01"), // Calm sideways weekend
    new Date("2025-05-13"), // No major events, flat market
    new Date("2025-07-26"), // Quiet Saturday, mid-cycle
    new Date("2024-10-02"), // Flat day in ranging market
    new Date("2023-08-21"), // Typical weekday, no impulse
]