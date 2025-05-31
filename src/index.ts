import { main } from "./order-book/main";

const DOWNLOAD_ONLY = false;
const FROM = "2023-06-17";
const TO = "2023-06-17";

console.log(`Processing from ${FROM} to ${TO}`);

main(DOWNLOAD_ONLY, FROM, TO, false, 5).catch(console.error);