# create by the team:
import ccxt
import time

class BitcoinTradingBot:
    """A Bitcoin trading bot that trades on multiple cryptocurrency exchanges using the ccxt library."""

    def __init__(self, exchange_id, api_key, api_secret, start_amount, stop_loss_percent, take_profit_percent):
        """
        Initializes a trading bot for a specific cryptocurrency exchange with given API credentials and trading parameters.
        Args:
            exchange_id (str): ID of the cryptocurrency exchange (e.g., 'binance').
            api_key (str): API key for the specified exchange.
            api_secret (str): API secret for the specified exchange.
            start_amount (float): Initial amount of Bitcoin to trade.
            stop_loss_percent (float): Stop loss threshold in percentage.
            take_profit_percent (float): Take profit threshold in percentage.
        """
        self.exchange = getattr(ccxt, exchange_id)({'apiKey': api_key, 'secret': api_secret})
        self.amount = start_amount
        self.market_pair = 'BTC/USD'  # This should be adjusted according to exchange market pair standards.
        self.stop_loss_threshold = stop_loss_percent / 100
        self.take_profit_threshold = take_profit_percent / 100

    def fetch_balance(self):
        """Returns the available balance of Bitcoin."""
        return self.exchange.fetch_balance()['free']['BTC']

    def fetch_current_price(self):
        """Returns the current market price of Bitcoin."""
        return self.exchange.fetch_ticker(self.market_pair)['last']

    def calculate_order_amount(self, price, order_type):
        """Calculate the amount of Bitcoin to buy or sell, adjusting for market conditions and wallet balance."""
        balance = self.fetch_balance()
        if order_type == 'buy':
            return min(self.amount, balance / price)
        return min(self.amount, balance)

    def place_order(self, side, price=None):
        """
        Places a buy or sell order.
        Args:
            side (str): 'buy' for buying, 'sell' for selling.
            price (float): Optional. Market price if not specified.
        """
        order_type = 'limit'
        price = price or self.fetch_current_price()
        amount = self.calculate_order_amount(price, side)
        
        if amount > 0:
            try:
                order_method = getattr(self.exchange, f'create_{order_type}_{side}_order')
                order = order_method(self.market_pair, amount, price)
                print(f'Placed {side} order: {amount} BTC at {price} USD')
            except ccxt.BaseError as e:
                print(f'Error placing {side} order: {str(e)}')
        else:
            print(f"Insufficient funds to place {side} order.")

    def monitor_market(self):
        """Monitors market conditions and adjusts trading strategy as per defined thresholds."""
        initial_price = self.fetch_current_price()
        
        try:
            while True:
                current_price = self.fetch_current_price()
                if current_price >= initial_price * (1 + self.take_profit_threshold):
                    self.place_order('sell')
                elif current_price <= initial_price * (1 - self.stop_loss_threshold):
                    self.place_order('buy')
                
                time.sleep(60)  # Sleep to rate limit the API usage.
        except KeyboardInterrupt:
            print("Trading interrupted by user.")

def main():
    bot = BitcoinTradingBot(exchange_id='binance', api_key='your_key', api_secret='your_secret',
                            start_amount=0.01, stop_loss_percent=10, take_profit_percent=10)
    bot.monitor_market()

if __name__ == '__main__':
    main()
