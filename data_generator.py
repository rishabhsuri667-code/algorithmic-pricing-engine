import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_data(num_properties=50, days=365):
    print(f"Generating synthetic data for {num_properties} properties over {days} days...")
    
    # 1. Generate Listings
    neighborhoods = ['Downtown', 'Uptown', 'Suburbs', 'Beachfront', 'Arts District']
    base_price_ranges = {
        'Downtown': (100, 250),
        'Uptown': (80, 180),
        'Suburbs': (60, 120),
        'Beachfront': (150, 400),
        'Arts District': (90, 200)
    }
    
    listings = []
    for i in range(1, num_properties + 1):
        neighborhood = np.random.choice(neighborhoods)
        min_p, max_p = base_price_ranges[neighborhood]
        base_price = np.random.randint(min_p, max_p)
        bedrooms = np.random.choice([1, 2, 3, 4], p=[0.4, 0.4, 0.15, 0.05])
        rating = round(np.random.uniform(3.5, 5.0), 2)
        
        listings.append({
            'property_id': f'PROP_{i:03d}',
            'name': f"{bedrooms}BR Cozy Stay in {neighborhood}",
            'neighborhood': neighborhood,
            'bedrooms': bedrooms,
            'base_price': base_price,
            'rating': rating
        })
        
    df_listings = pd.DataFrame(listings)
    
    # 2. Generate Daily Demand
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=d) for d in range(days)]
    
    # Generate common weather/event data for each day to be consistent across properties
    weather_conditions = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']
    daily_context = {}
    for date in dates:
        # Seasonality: summer is warmer, winter has snow
        month = date.month
        if month in [6, 7, 8]:
            probs = [0.7, 0.2, 0.1, 0.0]
        elif month in [12, 1, 2]:
            probs = [0.3, 0.3, 0.1, 0.3]
        else:
            probs = [0.4, 0.3, 0.3, 0.0]
            
        weather = np.random.choice(weather_conditions, p=probs)
        
        # Sparse events (e.g., concert, festival)
        is_event = np.random.choice([0, 1], p=[0.95, 0.05])
        
        # Holidays (simplified, just a random chance or specific days if we wanted, let's use random chance for simplicity but slightly higher in Nov/Dec)
        is_holiday = np.random.choice([0, 1], p=[0.98, 0.02])
        if month in [11, 12]:
            is_holiday = np.random.choice([0, 1], p=[0.90, 0.10])
            
        daily_context[date] = {
            'weather': weather,
            'is_event': is_event,
            'is_holiday': is_holiday
        }
    
    demand_records = []
    
    for idx, property in df_listings.iterrows():
        prop_id = property['property_id']
        base_price = property['base_price']
        neighborhood = property['neighborhood']
        
        for date in dates:
            context = daily_context[date]
            is_weekend = 1 if date.weekday() >= 5 else 0
            
            # The host decides on a price for this day. 
            # They might randomly hike it up or discount it slightly.
            price_multiplier = np.random.uniform(0.8, 1.5)
            
            # Smart hosts raise price on weekends/holidays/events
            if is_weekend: price_multiplier += np.random.uniform(0.0, 0.2)
            if context['is_holiday']: price_multiplier += np.random.uniform(0.1, 0.4)
            if context['is_event']: price_multiplier += np.random.uniform(0.1, 0.5)
            
            offered_price = round(base_price * price_multiplier, 2)
            
            # --- The "True" Market Demand Model ---
            # We calculate a 'willingness to pay' score.
            # If willingness > offered_price, it gets booked.
            
            # Base willingness is centered around the base price
            base_willingness = base_price * np.random.normal(1.05, 0.15) 
            
            # Modifiers
            if is_weekend: base_willingness *= 1.2
            if context['is_holiday']: base_willingness *= 1.4
            if context['is_event']: base_willingness *= 1.3
            
            if context['weather'] == 'Sunny': base_willingness *= 1.1
            elif context['weather'] == 'Rainy': base_willingness *= 0.9
            elif context['weather'] == 'Snowy' and neighborhood != 'Downtown': 
                # People might not want to travel to suburbs in snow
                base_willingness *= 0.8
                
            # Rating affects willingness slightly
            base_willingness *= (property['rating'] / 4.5)
            
            # Booked boolean
            booked = 1 if base_willingness >= offered_price else 0
            
            demand_records.append({
                'property_id': prop_id,
                'date': date.strftime('%Y-%m-%d'),
                'price': offered_price,
                'is_weekend': is_weekend,
                'is_holiday': context['is_holiday'],
                'is_event': context['is_event'],
                'weather': context['weather'],
                'booked': booked
            })
            
    df_demand = pd.DataFrame(demand_records)
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df_listings.to_csv('data/listings.csv', index=False)
    df_demand.to_csv('data/daily_demand.csv', index=False)
    
    print(f"Generated {len(df_listings)} listings and {len(df_demand)} demand records.")
    print("Files saved to 'data/' directory.")

if __name__ == "__main__":
    generate_data()
