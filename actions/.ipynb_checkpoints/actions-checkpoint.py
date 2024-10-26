
from typing import Any, Text, Dict, Tuple, List, Optional
from rasa_sdk import Action, Tracker
# from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from dateutil.relativedelta import relativedelta
import logging
logging.basicConfig(level=logging.INFO)
LOADED_DATASET = None

def load_sales_data(file_path='CLEANED_AIRHUB_NEW_DATA.xlsx'):
    global LOADED_DATASET
    if LOADED_DATASET is None:
        try:
            logging.info("Loading sales data from Excel file globally...")
            LOADED_DATASET = pd.read_excel(file_path)
            LOADED_DATASET['PurchaseDate'] = pd.to_datetime(LOADED_DATASET['PurchaseDate'], errors='coerce')
            LOADED_DATASET['salescount'] = pd.to_numeric(LOADED_DATASET['salescount'], errors='coerce')
            LOADED_DATASET['SellingPrice'] = pd.to_numeric(LOADED_DATASET['SellingPrice'], errors='coerce')
            if LOADED_DATASET is not None:
                logging.info("Dataset loaded successfully.")
                logging.info(f"Dataset shape: {LOADED_DATASET.shape}")
                if LOADED_DATASET.empty:
                    logging.warning("The dataset appears to be empty.")
                else:
                    logging.info("The dataset contains data.")
        except FileNotFoundError:
            logging.error("The sales data file could not be found.")
        except Exception as e:
            logging.error(f"An error occurred while loading the sales data: {str(e)}")
    else:
        logging.info("Dataset is already loaded in memory.")
            
load_sales_data()


months = { 'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3, 'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9, 'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12 }
months_reverse = {v: k for k, v in months.items()}

word_to_num = {
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4,
    'fifth': 5, 'sixth': 6, 'seventh': 7, 'eighth': 8,
    'ninth': 9, 'tenth': 10, 'eleventh': 11, 'twelfth': 12
}

def extract_months_from_text(text):
    # Regular expression to match numbers or words like '5', 'five', etc.
    pattern = r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?\b'
    
    # Search for the number of months in the text
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        # Extract the matched part
        month_str = match.group(1).lower()
        
        # Convert word-based numbers to digits, or just use the digit if it's already a number
        if month_str.isdigit():
            return int(month_str)
        else:
            return word_to_num.get(month_str, 0)  # Convert word to number, default to 0 if not found
    return 0

def get_month_range_from_text(text):
    # Extract the number of months
    num_months = extract_months_from_text(text)
    
    if num_months > 0:
        # Get the current date
        current_date = datetime.now()
        
        # Calculate the date X months ago
        past_date = current_date - relativedelta(months=num_months)
        
        # Return the range of months (past month/year to current month/year)
        month_range = [[past_date.month, past_date.year], [current_date.month, current_date.year]]
        return month_range, num_months
    else:
        return None
def extract_specific_date(user_message):
    # Regex pattern for extracting a specific date like "5 April 2024"
    date_pattern = r'\b(\d{1,2})\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s*(\d{4})\b'
    match = re.search(date_pattern, user_message, re.IGNORECASE)
    if match:
        day, month_name, year = match.groups()
        month = months_reverse[month_name.lower()]
        specific_date = pd.to_datetime(f"{year}-{month}-{day}", format="%Y-%m-%d")
        return specific_date.date()
    return None

# def extract_months(text):
#     # Regular expression to find month names or abbreviations (case-insensitive)
#     pattern = r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b'
    
#     # Find all matches of month names (case-insensitive)
#     matches = re.findall(pattern, text, re.IGNORECASE)
    
#     # Convert matched month names to corresponding digits
#     month_digits = [months[match.lower()] for match in matches]
    
#     return month_digits

def extract_months(text):
    """Extracts month numbers from user input based on month names or numeric/ordinal representations."""
    
    # Regular expression to find month names or abbreviations (case-insensitive)
    month_pattern = r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b'
    
    # Regular expression to find numeric months (1-12)
    numeric_pattern = r'\b(1[0-2]|[1-9])\b'
    
    # Regular expression to find ordinal months (first to twelfth)
    ordinal_pattern = r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\b'
    
    # Find all matches of month names
    matches = re.findall(month_pattern, text, re.IGNORECASE)
    
    # Convert matched month names to corresponding digits
    month_digits = [months[match.lower()] for match in matches]

    # Check for numeric months
    numeric_match = re.search(numeric_pattern, text)
    if numeric_match:
        month_digits.append(int(numeric_match.group(0)))

    # Check for ordinal months
    ordinal_match = re.search(ordinal_pattern, text)
    if ordinal_match:
        month_digits.append(word_to_num.get(ordinal_match.group(0), None))

    return list(set(month_digits))

def extract_years(text):
    # Regular expression to match years in YYYY format without capturing the century separately
    pattern = r'\b(?:19|20)\d{2}\b'
    
    # Find all matches of the pattern
    years = re.findall(pattern, text)
    
    return [int(year) for year in years]


def extract_month_year(text):    
    pattern = r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*(?:of\s*)?(\d{4})\b'
    
    # Find all matches of the pattern (month and year)
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    # Convert matched month names to corresponding digits and pair them with the year as arrays
    month_year_pairs = [[months[month.lower()], int(year)] for month, year in matches]
    
    return month_year_pairs

def extract_date_range(user_message):
    # Regex pattern for extracting a date range like "from 2 April 2023 to 30 June 2024"
    range_pattern = r'from\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s*(\d{4})\s+to\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s*(\d{4})'
    match = re.search(range_pattern, user_message, re.IGNORECASE)
    if match:
        start_month_name, start_year,end_month_name, end_year = match.groups()
        start_month = months_reverse[start_month_name.lower()]
        end_month = months_reverse[end_month_name.lower()]
        start_date = pd.to_datetime(f"{start_year}-{start_month}-{start_day}", format='%B %Y')
        end_date = pd.to_datetime(f"{end_year}-{end_month}", format="%B %Y")
        return start_date.date(), end_date.date()
    return None

def convert_to_datetime(self, date_str: str) -> datetime:
    """Converts a date string to a datetime object."""
        
    # Normalize the string for parsing
    date_str = date_str.replace("st", "").replace("nd", "").replace("rd", "").replace("th", "")
        
    try:
        return datetime.strptime(date_str.strip(), "%d %B %Y")
    except ValueError:
        # Handle month-only strings (e.g., "August 2022")
        return datetime.strptime(date_str.strip(), "%B %Y")

def calculate_country_sales(df, country):
    # Filter the data for the given country
    country_data = df[df['countryname'].str.lower() == country.lower()]
    
    # Calculate the sales count and total revenue
    sales_count = country_data['SellingPrice'].count()
    total_revenue = country_data['SellingPrice'].sum()
    
    return sales_count, total_revenue
    
def calculate_country_sales_by_year(df, country, year):
    country_data = df[(df['countryname'].str.lower() == country.lower()) & (df['PurchaseDate'].dt.year == year)]
    sales_count = country_data['SellingPrice'].count()
    total_revenue = country_data['SellingPrice'].sum()
    return sales_count, total_revenue

def calculate_country_sales_by_month_year(df, country, month, year):
    country_data = df[(df['countryname'].str.lower() == country.lower()) & 
                      (df['PurchaseDate'].dt.month == month) &
                      (df['PurchaseDate'].dt.year == year)]
    sales_count = country_data['SellingPrice'].count()
    total_revenue = country_data['SellingPrice'].sum()
    return sales_count, total_revenue

def extract_country_from_text(text, country_names):
    text_lower = text.lower()    
    matched_countries = [country for country in country_names if country.lower() in text_lower]
    return matched_countries


def extract_country(message: str, valid_countries: List[str]) -> List[str]:
    # Compile a regex pattern to match valid country names
    country_pattern = r'\b(?:' + '|'.join(map(re.escape, valid_countries)) + r')\b'
    
    # Find all matches for countries in the user message
    found_countries = re.findall(country_pattern, message, re.IGNORECASE)

    # Return the unique list of found countries
    return list(set(found_countries))[:2]  # Return only the first two unique matches
def calculate_region_sales(df, region):
    """Calculates total sales and revenue for the given region."""
    region_sales = df[df['regionname'].str.lower() == region.lower()]
    total_sales = region_sales['SellingPrice'].count()
    total_revenue = region_sales['SellingPrice'].sum()
    return total_sales, total_revenue

def calculate_region_sales_by_month_year(df, region, month, year):
    """Calculates total sales and revenue for a region for a specific month and year."""
    region_sales = df[(df['regionname'].str.lower() == region.lower()) & (df['PurchaseDate'].dt.month == month) &
                      (df['PurchaseDate'].dt.year == year)]
                      
    total_sales = region_sales['SellingPrice'].count()
    total_revenue = region_sales['SellingPrice'].sum()
    return total_sales, total_revenue

def calculate_region_sales_by_year(df, region, year):
    """Calculates total sales and revenue for a region for a specific year."""
    region_sales = df[(df['regionname'].str.lower() == region.lower()) & (df['PurchaseDate'].dt.year == year)]
    total_sales = region_sales['SellingPrice'].count()
    total_revenue = region_sales['SellingPrice'].sum()
    return total_sales, total_revenue

def calculate_total_region_sales(df, region):
    """Calculates total sales and revenue for the given region (all years and months)."""
    return calculate_region_sales(df, region)


def calculate_planname_sales_by_month_year(df, planname, month, year):
    if 'month' not in df.columns or 'year' not in df.columns:
        # Convert 'PurchaseDate' (or equivalent) to datetime and extract month and year
        df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
        df['month'] = df['PurchaseDate'].dt.month
        df['year'] = df['PurchaseDate'].dt.year
    filtered_df = df[(df['PlanName'] == planname) & (df['month'] == month) & (df['year'] == year)]
    sales_count = filtered_df['SellingPrice'].count()
    total_revenue =  filtered_df['SellingPrice'].sum()
    return sales_count, total_revenue

def calculate_planname_sales_by_year(df, planname, year):
    if 'month' not in df.columns or 'year' not in df.columns:
        # Convert 'PurchaseDate' (or equivalent) to datetime and extract month and year
        df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
        df['month'] = df['PurchaseDate'].dt.month
        df['year'] = df['PurchaseDate'].dt.year
    filtered_df = df[(df['PlanName'] == planname) & (df['year'] == year)]
    sales_count = filtered_df['SellingPrice'].count()
    total_revenue = filtered_df['SellingPrice'].sum()
    return sales_count, total_revenue

def calculate_total_planname_sales(df, planname):
    filtered_df = df[df['PlanName'] == planname]
    sales_count = filtered_df['SellingPrice'].count()
    total_revenue =  filtered_df['SellingPrice'].sum()
    return sales_count, total_revenue


def calculate_top_sales_plans(df, year=None, month=None):
    # Implement logic to calculate top selling plans
    # Example:
    if year:
        df = df[df['PurchaseDate'].dt.year == year]
    if month:
        df = df[df['PurchaseDate'].dt.month == month]
    return df.groupby('PlanName').agg(SalesCount=('SellingPrice', 'count'), TotalRevenue=('SellingPrice', 'sum')).nlargest(10, 'SalesCount')

def calculate_least_sales_plans(df, year=None, month=None):
    # Implement logic to calculate least selling plans
    if year:
        df = df[df['PurchaseDate'].dt.year == year]
    if month:
        df = df[df['PurchaseDate'].dt.month == month]
    return df.groupby('PlanName').agg(SalesCount=('SellingPrice', 'count'), TotalRevenue=('SellingPrice', 'sum')).nsmallest(10, 'SalesCount')





#######################TOTALSALES#############################################################################################

class ActionGetTotalSales(Action):
    def name(self) -> Text:
        return "action_get_total_sales"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            logging.info("Running ActionGetTotalSales...")

            global LOADED_DATASET

            if LOADED_DATASET is None:
                LOADED_DATASET = load_sales_data()
                if LOADED_DATASET is None:
                    dispatcher.utter_message(text="Error: Sales data could not be loaded. Please check the dataset source.")
                    return []

            user_message = tracker.latest_message.get('text')
            df = LOADED_DATASET.copy()
            if df.empty or df['SellingPrice'].isnull().all():
                dispatcher.utter_message(text="Error: The sales data is empty or invalid.")
                return []

            years = extract_years(user_message)
            months_extracted = extract_months(user_message)
            month_year_pairs = extract_month_year(user_message)
            specific_date = extract_specific_date(user_message)
            date_range = extract_date_range(user_message)
            total_sales_count = 0
            total_sales_price = 0.0

            if specific_date:
                try:
                    specific_date = pd.to_datetime(specific_date, errors='coerce').date()
                    if pd.isna(specific_date):
                        dispatcher.utter_message(text="Error: The provided date is invalid. Please provide a valid date.")
                        return []
                    
                    daily_sales_count = df[df['PurchaseDate'] == specific_date]['SellingPrice'].count()
                    daily_sales_price = df[df['PurchaseDate'] == specific_date]['SellingPrice'].sum()

                    if daily_sales_count > 0:
                        dispatcher.utter_message(text=f"The total sales for {specific_date.strftime('%d %B %Y')} is {daily_sales_count} with a sales price of ${daily_sales_price:.2f}.")
                    else:
                        dispatcher.utter_message(text=f"No sales were recorded on {specific_date.strftime('%d %B %Y')}.")
                except Exception as e:
                    dispatcher.utter_message(text=f"Error occurred while processing specific date sales: {str(e)}")
                return []

            elif date_range:
                try:
                    start_date, end_date = date_range
                    start_date = pd.to_datetime(start_date, errors='coerce')
                    end_date = pd.to_datetime(end_date, errors='coerce')
                    if pd.isna(start_date) or pd.isna(end_date):
                        dispatcher.utter_message(text="Error: Please provide a valid date range in the format 'month year to month year'.")
                        return []

                    start_date = start_date.replace(day=1)
                    end_date = end_date.replace(day=1) + pd.offsets.MonthEnd(0)
                    if start_date <= end_date:
                        filtered_df = df[(df['PurchaseDate'] >= start_date) & (df['PurchaseDate'] <= end_date)]

                        range_sales_count = filtered_df['SellingPrice'].count()
                        range_sales_price = filtered_df['SellingPrice'].sum()

                        if range_sales_count > 0:
                            dispatcher.utter_message(text=f"The total sales from {start_date.strftime('%B %Y')} to {end_date.strftime('%B %Y')} is {range_sales_count} with a sales price of ${range_sales_price:.2f}.")
                        else:
                            dispatcher.utter_message(text=f"No sales were recorded from {start_date.strftime('%B %Y')} to {end_date.strftime('%B %Y')}.")
                    else:
                        dispatcher.utter_message(text="Error: The end date must be after the start date. Please provide a valid date range.")
                except Exception as e:
                    dispatcher.utter_message(text=f"Error occurred while processing date range sales: {str(e)}")
                return []

            if month_year_pairs:
                try:
                    total_sales_count = total_sales_price = 0.0
                    for month, year in month_year_pairs:
                        monthly_sales_count = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)]['SellingPrice'].count()
                        monthly_sales_price = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)]['SellingPrice'].sum()
                        total_sales_count += monthly_sales_count
                        total_sales_price += monthly_sales_price

                        dispatcher.utter_message(text=f"The total sales count for {months_reverse[month]} {year} is {monthly_sales_count} and sales price is ${monthly_sales_price:.2f}.")
                except Exception as e:
                    dispatcher.utter_message(text=f"Error occurred while processing monthly sales by year: {str(e)}")
                return []

            if years:
                try:
                    total_sales_count = df[df['PurchaseDate'].dt.year.isin(years)]['SellingPrice'].count()
                    total_sales_price = df[df['PurchaseDate'].dt.year.isin(years)]['SellingPrice'].sum()
                    
                    if len(years) == 1:
                        dispatcher.utter_message(text=f"The total sales count for {years[0]} is {total_sales_count} and sales price is ${total_sales_price:.2f}.")
                    else:
                        years_str = ', '.join(map(str, years))
                        dispatcher.utter_message(text=f"Total sales count for {years_str} is {total_sales_count} and total sales price is ${total_sales_price:.2f}.")
                except Exception as e:
                    dispatcher.utter_message(text=f"Error occurred while processing annual sales: {str(e)}")
                return []

            if months_extracted:
                try:
                    current_year = pd.to_datetime('today').year
                    for month in months_extracted:
                        monthly_sales_count = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)]['SellingPrice'].count()
                        monthly_sales_price = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)]['SellingPrice'].sum()
                        total_sales_count += monthly_sales_count
                        total_sales_price += monthly_sales_price
                        
                        dispatcher.utter_message(text=f"The total sales count for {months_reverse[month]} {current_year} is {monthly_sales_count} and sales price is ${monthly_sales_price:.2f}.")
                except Exception as e:
                    dispatcher.utter_message(text=f"Error occurred while processing monthly sales: {str(e)}")
                return []

            total_sales_count, total_sales_price = self.calculate_total_sales(df)
            dispatcher.utter_message(text=f"The overall total sales count is {total_sales_count} with a total sales price of ${total_sales_price:.2f}.")

        except Exception as e:
            dispatcher.utter_message(text=f"An error occurred in processing your request: {str(e)}")
        
        return []

    def calculate_total_sales(self, df: pd.DataFrame) -> Tuple[int, float]:
        total_sales_count = df['salescount'].sum() if 'salescount' in df.columns else 0
        total_sales_price = df['SellingPrice'].sum() if 'SellingPrice' in df.columns else 0.0
        return total_sales_count, total_sales_price

###############################################COMPARESALES################################################################################

class ActionCompareSales(Action):
    def name(self) -> str:
        return "action_compare_sales"

    def run(self, dispatcher, tracker, domain) -> list[Dict[Text, Any]]:
        try:
            logging.info("Running ActionCompareSales...")
            global LOADED_DATASET

            if LOADED_DATASET is None:
                LOADED_DATASET = load_sales_data()
                if LOADED_DATASET is None:
                    dispatcher.utter_message(text="Sales data could not be loaded. Please check the dataset source.")
                    return []

            user_message = tracker.latest_message.get('text')
            if not user_message:
                dispatcher.utter_message(text="I didn't receive any message for comparison. Please specify a time range.")
                return []

            df = LOADED_DATASET.copy()

            # Ensure required column exists in dataset
            if 'PurchaseDate' not in df.columns or 'SellingPrice' not in df.columns:
                dispatcher.utter_message(text="Required columns ('PurchaseDate', 'SellingPrice') are missing in the dataset.")
                return []

            # Check if 'PurchaseDate' is in datetime format, if not try to convert
            if not pd.api.types.is_datetime64_any_dtype(df['PurchaseDate']):
                try:
                    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
                except Exception as e:
                    dispatcher.utter_message(text="Failed to parse 'PurchaseDate' as datetime. Please check the date format.")
                    logging.error(f"Date parsing error: {e}")
                    return []

            # Define patterns for month and year comparison
            month_pattern = r"(\w+ \d{4}) to (\w+ \d{4})"
            year_pattern = r"(\d{4}) to (\d{4})"

            # Check for month comparison
            month_matches = re.findall(month_pattern, user_message)
            if month_matches:
                try:
                    month1, month2 = month_matches[0]
                    start_date_1 = datetime.strptime(month1, "%B %Y").replace(day=1)
                    start_date_2 = datetime.strptime(month2, "%B %Y").replace(day=1)

                    logging.info(f"Comparing sales for {month1} and {month2}...")
                    
                    sales_count_1 = df[(df['PurchaseDate'].dt.month == start_date_1.month) & 
                                        (df['PurchaseDate'].dt.year == start_date_1.year)]['SellingPrice'].count()
                    sales_price_1 = df[(df['PurchaseDate'].dt.month == start_date_1.month) & 
                                        (df['PurchaseDate'].dt.year == start_date_1.year)]['SellingPrice'].sum()
                    
                    sales_count_2 = df[(df['PurchaseDate'].dt.month == start_date_2.month) & 
                                        (df['PurchaseDate'].dt.year == start_date_2.year)]['SellingPrice'].count()
                    sales_price_2 = df[(df['PurchaseDate'].dt.month == start_date_2.month) & 
                                        (df['PurchaseDate'].dt.year == start_date_2.year)]['SellingPrice'].sum()

                    comparison_text = (
                         f"Sales Comparison between {month1} and {month2}:\n\n"
                         f"• {month1}:\n"
                         f"   - Total Sales: {sales_count_1} sales\n"
                         f"   - Sales Revenue: ${sales_price_1:.2f}\n\n"
                         f"• {month2}:\n"
                         f"   - Total Sales: {sales_count_2} sales\n"
                         f"   - Sales Revenue: ${sales_price_2:.2f}\n\n"
                    )

                    count_difference = sales_count_1 - sales_count_2
                    price_difference = sales_price_1 - sales_price_2

                    if count_difference > 0:
                        comparison_text += f"Difference in sales: {month1} had {abs(count_difference)} more sales than {month2}.\n"
                    elif count_difference < 0:
                        comparison_text += f"Difference in sales: {month2} had {abs(count_difference)} more sales than {month1}.\n"
                    else:
                        comparison_text += " Both months had the same number of sales.\n"

                    if price_difference > 0:
                        comparison_text += f"Difference in revenue: {month1} generated ${abs(price_difference):.2f} more in sales revenue than {month2}."
                    elif price_difference < 0:
                        comparison_text += f"Difference in revenue: {month2} generated ${abs(price_difference):.2f} more in sales revenue than {month1}."
                    else:
                        comparison_text += " Both months generated the same sales revenue."

                    dispatcher.utter_message(text=comparison_text)
                except ValueError as ve:
                    dispatcher.utter_message(text="Invalid date format for month comparison. Please use 'Month Year' format.")
                    logging.error(f"Date parsing error for month comparison: {ve}")
                return []

            # Check for year comparison
            year_matches = re.findall(year_pattern, user_message)
            if year_matches:
                try:
                    year1, year2 = year_matches[0]
                    sales_count_1 = df[df['PurchaseDate'].dt.year == int(year1)]['SellingPrice'].count()
                    sales_price_1 = df[df['PurchaseDate'].dt.year == int(year1)]['SellingPrice'].sum()
                    
                    sales_count_2 = df[df['PurchaseDate'].dt.year == int(year2)]['SellingPrice'].count()
                    sales_price_2 = df[df['PurchaseDate'].dt.year == int(year2)]['SellingPrice'].sum()

                    comparison_text = (
                        f"Sales Comparison between {year1} and {year2}:\n\n"
                        f"• {year1}:\n"
                        f"   - Total Sales: {sales_count_1} sales\n"
                        f"   - Total Revenue: ${sales_price_1:.2f}\n\n"
                        f"• {year2}:\n"
                        f"   - Total Sales: {sales_count_2} sales\n"
                        f"   - Total Revenue: ${sales_price_2:.2f}\n\n"
                    )
                    
                    count_difference = sales_count_1 - sales_count_2
                    price_difference = sales_price_1 - sales_price_2

                    if count_difference > 0:
                        comparison_text += f"**Sales Count Difference:** {year1} had {abs(count_difference)} more sales than {year2}.\n"
                    elif count_difference < 0:
                        comparison_text += f"**Sales Count Difference:** {year2} had {abs(count_difference)} more sales than {year1}.\n"
                    else:
                        comparison_text += " Both years had the same number of sales."

                    if price_difference > 0:
                        comparison_text += f"**Revenue Difference:** {year1} generated ${abs(price_difference):.2f} more in sales revenue than {year2}."
                    elif price_difference < 0:
                        comparison_text += f"**Revenue Difference:** {year2} generated ${abs(price_difference):.2f} more in sales revenue than {year1}."
                    else:
                        comparison_text += " Both years generated the same sales revenue."

                    dispatcher.utter_message(text=comparison_text)
                except ValueError as ve:
                    dispatcher.utter_message(text="Invalid date format for year comparison. Please use 'YYYY' format.")
                    logging.error(f"Date parsing error for year comparison: {ve}")
                return []

            dispatcher.utter_message(text="Please provide a valid comparison in the format 'month year to month year' or 'year to year'.")
        
        except Exception as e:
            dispatcher.utter_message(text="An error occurred while processing your request. Please try again.")
            logging.error(f"Unexpected error in ActionCompareSales: {e}")
        
        return []




##############################################################salesforcountry###########################################################################

class ActionCountrySales(Action):
    def name(self) -> str:
        return "action_country_sales"
   
    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        logging.info("Running ActionCountrySales...")
        global LOADED_DATASET
        
        try:
            # Load dataset if not already loaded
            if LOADED_DATASET is None:
                LOADED_DATASET = load_sales_data()
                if LOADED_DATASET is None:
                    dispatcher.utter_message(text="Sales data could not be loaded. Please check the dataset source.")
                    logging.error("Sales data could not be loaded.")
                    return []
            
            # Get user message and extract country name (assume country slot or from message)
            user_message = tracker.latest_message.get('text')
            df = LOADED_DATASET.copy()
            country = next(tracker.get_latest_entity_values('country'), None)
            logging.info(f"Extracted country: {country}")
            valid_countries = df['countryname'].unique().tolist()
            valid_countries = [country_name.lower() for country_name in valid_countries]
            
            # Validate country name
            if not country:
                dispatcher.utter_message(text="Please provide a valid country name.")
                logging.warning("Country not provided by the user.")
                return []
            else:
                # Normalize the country name to lower case for comparison
                normalized_country = country.lower()
                if normalized_country not in valid_countries:
                    # Check if the country name contains two words and validate
                    matched_countries = [c for c in valid_countries if normalized_country in c]
                    if matched_countries:
                        country = matched_countries[0]  # Use the first matched country
                    else:
                        dispatcher.utter_message(text=f"Sorry, we do not have sales data for {country}. Please provide another country.")
                        logging.warning(f"Country {country} not found in the dataset.")
                        return []
            
            # Extract years, months, and month-year pairs from user message
            years = extract_years(user_message)
            months_extracted = extract_months(user_message)
            month_year_pairs = extract_month_year(user_message)
            logging.info(f"Processing sales data for country: {country}")
            
            response_message = ""
            
            # Handle month-year pairs
            if month_year_pairs:
                logging.info("Processing request for total sales with specific month and year...")
                for month, year in month_year_pairs:
                    sales_count, total_revenue = calculate_country_sales_by_month_year(df, country, month, year)
                    response_message += (f"In {months_reverse.get(month, month).capitalize()} {year}, "
                                         f"{country} recorded {sales_count} sales, generating a total revenue of ${total_revenue:,.2f}.\n")
                    
            # Handle year-only input
            elif years:
                for year in years:
                    sales_count, total_revenue = calculate_country_sales_by_year(df, country, year)
                    response_message += (f"In {year}, {country} recorded {sales_count} sales, generating a total revenue of ${total_revenue:,.2f}.\n")
            
            # Handle month-only input, assuming current year
            elif months_extracted:
                current_year = datetime.now().year
                for month in months_extracted:
                    sales_count, total_revenue = calculate_country_sales_by_month_year(df, country, month, current_year)
                    response_message += (f"In {months_reverse.get(month, month).capitalize()} {current_year}, "
                                         f"{country} recorded {sales_count} sales, generating a total revenue of ${total_revenue:,.2f}.\n")
            
            # If no specific month or year is mentioned, return total sales for the country
            else:
                sales_count, total_revenue = calculate_country_sales(df, country)
                response_message = (f"In {country}, there have been a total of {sales_count} sales, "
                                    f"generating a total revenue of ${total_revenue:,.2f}.")
            
            # Send response to user
            dispatcher.utter_message(text=response_message)
        
        except KeyError as e:
            logging.error(f"KeyError: {e}")
            dispatcher.utter_message(text="An error occurred while processing the sales data. Please try again.")
        except ValueError as e:
            logging.error(f"ValueError: {e}")
            dispatcher.utter_message(text="There was an issue with the input data format. Please specify the dates correctly.")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            dispatcher.utter_message(text="An unexpected error occurred. Please try again later.")
        
        return []





##########################################################################################################################PLANRELATEDSALES#################

class ActionPlanNameByCountry(Action):
    def name(self) -> str:
        return "action_planname_by_country"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        logging.info("Running ActionPlanNameByCountry...")
        global LOADED_DATASET

        try:
            # Load the dataset if not already loaded
            if LOADED_DATASET is None:
                LOADED_DATASET = load_sales_data()
                if LOADED_DATASET is None:
                    dispatcher.utter_message(text="Sales data could not be loaded. Please check the dataset source.")
                    logging.error("Failed to load sales data.")
                    return []

            # Get user message and extract entities
            user_message = tracker.latest_message.get('text')
            df = LOADED_DATASET.copy()
            country = next(tracker.get_latest_entity_values('countryname'), None)
            planname = next(tracker.get_latest_entity_values('planname'), None)

            # Check for a valid country input
            if not country:
                dispatcher.utter_message(text="Please provide a valid country name.")
                logging.warning("Country name not provided by the user.")
                return []
            elif country not in df['countryname'].values:
                dispatcher.utter_message(text=f"Sorry, we do not have data for {country}. Please try another country.")
                logging.warning(f"Country {country} not found in the dataset.")
                return []

            # Extract plans for the specified country
            country_plans = df[df['countryname'] == country]['PlanName'].unique()
            if len(country_plans) == 0:
                dispatcher.utter_message(text=f"No plans available for {country}.")
                logging.info(f"No plans found for country: {country}")
                return []

            # Extract years, months, and month-year pairs from user message
            years = extract_years(user_message)
            months_extracted = extract_months(user_message)
            month_year_pairs = extract_month_year(user_message)
            logging.info(f"Processing sales data for country: {country} and plan: {planname}")
            response_message = ""

            # Helper function to format sales and revenue
            def format_sales_data(planname, sales_count, total_revenue):
                return (f"Plan: {planname}\n"
                        f"Total Sales: {sales_count}\n"
                        f"Revenue: ${total_revenue:,.2f}")

            # Generate response based on provided filters
            if month_year_pairs:
                for month, year in month_year_pairs:
                    response_message += f"📅 Sales Overview for {months_reverse.get(month, month).capitalize()} {year} ({country} Plans):\n\n"
                    for plan in country_plans:
                        sales_count, total_revenue = calculate_planname_sales_by_month_year(df, plan, month, year)
                        if sales_count > 0 and total_revenue > 0:
                            response_message += format_sales_data(plan, sales_count, total_revenue) + "\n\n"

            elif years:
                for year in years:
                    response_message += f"📅 Sales Overview for {year} ({country} Plans):\n\n"
                    for plan in country_plans:
                        sales_count, total_revenue = calculate_planname_sales_by_year(df, plan, year)
                        if sales_count > 0 or total_revenue > 0:
                            response_message += format_sales_data(plan, sales_count, total_revenue) + "\n\n"

            elif months_extracted:
                current_year = datetime.now().year
                for month in months_extracted:
                    response_message += f"📅 Sales Overview for {months_reverse.get(month, month).capitalize()} {current_year} ({country} Plans):\n\n"
                    for plan in country_plans:
                        sales_count, total_revenue = calculate_planname_sales_by_month_year(df, plan, month, current_year)
                        if sales_count > 0 or total_revenue > 0:
                            response_message += format_sales_data(plan, sales_count, total_revenue) + "\n\n"

            else:
                response_message += f"📊 Total Sales Overview for {country} Plans:\n\n"
                for plan in country_plans:
                    sales_count, total_revenue = calculate_total_planname_sales(df, plan)
                    if sales_count > 0 or total_revenue > 0:
                        response_message += format_sales_data(plan, sales_count, total_revenue) + "\n\n"

            # Check if no data was found
            if not response_message.strip():
                dispatcher.utter_message(text=f"No sales data found for the specified criteria in {country}.")
                logging.info("No sales data found for specified criteria.")
                return []

            # Send the formatted message
            dispatcher.utter_message(text=response_message)

        except KeyError as e:
            logging.error(f"KeyError: {e}")
            dispatcher.utter_message(text="An error occurred while processing the sales data. Please try again.")
        except ValueError as e:
            logging.error(f"ValueError: {e}")
            dispatcher.utter_message(text="There was an issue with the input data format. Please specify the dates correctly.")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            dispatcher.utter_message(text="An unexpected error occurred. Please try again later.")
        
        return []


#################################################################################################################active plans and country name##############


class ActionGetActiveAndInactivePlans(Action):
    def name(self) -> str:
        return "action_get_active_inactive_plans"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict) -> list:
        logging.info("Running ActionActivePlans...")
        global LOADED_DATASET

        try:
            # Load the dataset if not already loaded
            if LOADED_DATASET is None:
                LOADED_DATASET = load_sales_data()
                if LOADED_DATASET is None:
                    dispatcher.utter_message(text="Sales data could not be loaded. Please check the dataset source.")
                    logging.error("Sales data could not be loaded.")
                    return []

            user_message = tracker.latest_message.get('text')
            df = LOADED_DATASET.copy()

            # Ensure 'PurchaseDate' can be converted to datetime
            try:
                df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
            except Exception as e:
                dispatcher.utter_message(text="Error parsing purchase dates. Please check the date format.")
                logging.error(f"Date parsing error: {e}")
                return []

            # Get the current year dynamically
            try:
                current_year = datetime.now().year
            except Exception as e:
                dispatcher.utter_message(text="Could not retrieve the current year.")
                logging.error(f"Error fetching current year: {e}")
                return []

            # Identify active and inactive plans
            active_plans_current_year = df[df['PurchaseDate'].dt.year == current_year]['PlanName'].unique()
            past_plans = df[df['PurchaseDate'].dt.year.isin([2021, 2022, 2023])]['PlanName'].unique()
            active_plans = list(active_plans_current_year)
            inactive_plans = list(set(past_plans) - set(active_plans))
            plans_count = len(active_plans)
            inactive_plans_count = len(inactive_plans)

            # Generate response
            if plans_count == 0 and inactive_plans_count == 0:
                dispatcher.utter_message(text="No active or inactive plans found for the specified criteria.")
                logging.info("No plans found for the specified criteria.")
                return []

            response = f"Total Active Plans: {plans_count}\n\n"
            response += "HERE ARE THE ACTIVE PLANS:\n" + "\n".join(f"- {plan}" for plan in active_plans) + "\n\n"
            response += f"Total Inactive Plans: {inactive_plans_count}\n\n"
            response += "HERE ARE THE INACTIVE PLANS:\n" + "\n".join(f"- {plan}" for plan in inactive_plans)

            dispatcher.utter_message(text=response)
        
        except Exception as e:
            dispatcher.utter_message(text="An unexpected error occurred. Please try again later.")
            logging.error(f"Unexpected error in ActionGetActiveAndInactivePlans: {e}")
        
        return []


class ActionGetActiveAndInactiveCountries(Action):
    def name(self) -> str:
        return "action_get_active_inactive_countries"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict) -> list:
        logging.info("Running ActionActiveCountry...")
        global LOADED_DATASET

        try:
            # Load the dataset if not already loaded
            if LOADED_DATASET is None:
                LOADED_DATASET = load_sales_data()
                if LOADED_DATASET is None:
                    dispatcher.utter_message(text="Sales data could not be loaded. Please check the dataset source.")
                    logging.error("Sales data could not be loaded.")
                    return []

            user_message = tracker.latest_message.get('text')
            df = LOADED_DATASET.copy()

            # Ensure 'PurchaseDate' can be converted to datetime
            try:
                df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
            except Exception as e:
                dispatcher.utter_message(text="Error parsing purchase dates. Please check the date format.")
                logging.error(f"Date parsing error: {e}")
                return []

            # Get the current year dynamically
            try:
                current_year = datetime.now().year
            except Exception as e:
                dispatcher.utter_message(text="Could not retrieve the current year.")
                logging.error(f"Error fetching current year: {e}")
                return []

            # Identify active and inactive countries
            active_countries_current_year = df[df['PurchaseDate'].dt.year == current_year]['countryname'].unique()
            past_countries = df[df['PurchaseDate'].dt.year.isin([2021, 2022, 2023])]['countryname'].unique()
            active_countries = list(active_countries_current_year)
            inactive_countries = list(set(past_countries) - set(active_countries))
            countries_count = len(active_countries)
            inactive_countries_count = len(inactive_countries)

            # Generate response
            if countries_count == 0 and inactive_countries_count == 0:
                dispatcher.utter_message(text="No active or inactive countries found for the specified criteria.")
                logging.info("No countries found for the specified criteria.")
                return []

            response = f"Total Active Countries: {countries_count}\n\n"
            response += "HERE ARE THE ACTIVE COUNTRIES:\n" + "\n".join(f"- {country}" for country in active_countries) + "\n\n"
            response += f"Total Inactive Countries: {inactive_countries_count}\n\n"
            response += "HERE ARE THE INACTIVE COUNTRIES:\n" + "\n".join(f"- {country}" for country in inactive_countries)

            dispatcher.utter_message(text=response)
        
        except Exception as e:
            dispatcher.utter_message(text="An unexpected error occurred. Please try again later.")
            logging.error(f"Unexpected error in ActionGetActiveAndInactiveCountries: {e}")
        
        return []



#############################################################################################################top and lowest sales plan############

from datetime import datetime
import logging
from typing import List

class ActionTopPlansSales(Action):

    def name(self) -> str:
        return "action_top_plans_sales"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        logging.info("Running ActionTopPlansSales...")
        
        global LOADED_DATASET

        # Load the dataset if not already loaded
        if LOADED_DATASET is None:
            LOADED_DATASET = load_sales_data()
            if LOADED_DATASET is None:
                dispatcher.utter_message(text="Sales data could not be loaded. Please check the dataset source.")
                return []

        user_message = tracker.latest_message.get('text')
        df = LOADED_DATASET.copy()

        # Extract year and month from user message
        years = extract_years(user_message)
        months_extracted = extract_months(user_message)
        month_year_pairs = extract_month_year(user_message)

        logging.info(f"Processing top plans sales data based on user request: {user_message}")
        response_message = ""

        # Helper function to format the output
        def format_sales_data(planname, sales_count, total_revenue):
            return (f"Plan: {planname}\n "
                    f"Total Sales: {sales_count}\n "  
                    f"Revenue: ${total_revenue:,.2f}")

        try:
            # Determine the response based on user input
            if month_year_pairs:
                month, year = month_year_pairs[0]
                response_message += f"📈 Top 10 Plans in {months_reverse[month].capitalize()} {year}:\n\n"
                top_plans = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nlargest(10, 'total_sales').reset_index()
                if top_plans.empty:
                    response_message += "No data available for the specified month and year.\n"
                else:
                    for _, row in top_plans.iterrows():
                        response_message += format_sales_data(row['PlanName'], row['total_sales'], row['total_revenue']) + "\n\n"

            elif years:
                for year in years:
                    response_message += f"📈 Top 10 Plans in {year}:\n\n"
                    top_plans = df[df['PurchaseDate'].dt.year == year].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nlargest(10, 'total_sales').reset_index()
                    if top_plans.empty:
                        response_message += f"No data available for the year {year}.\n"
                    else:
                        for _, row in top_plans.iterrows():
                            response_message += format_sales_data(row['PlanName'], row['total_sales'], row['total_revenue']) + "\n\n"

            elif months_extracted:
                current_year = datetime.now().year
                for month in months_extracted:
                    response_message += f"📈 Top 10 Plans in {months_reverse[month].capitalize()} {current_year}:\n\n"
                    top_plans = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nlargest(10, 'total_sales').reset_index()
                    if top_plans.empty:
                        response_message += f"No data available for the month {months_reverse[month]} {current_year}.\n"
                    else:
                        for _, row in top_plans.iterrows():
                            response_message += format_sales_data(row['PlanName'], row['total_sales'], row['total_revenue']) + "\n\n"

            else:
                response_message += "📈 Top 10 Highest Sales Plans Overall:\n\n"
                top_plans = df.groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nlargest(10, 'total_sales').reset_index()
                if top_plans.empty:
                    response_message += "No data available overall.\n"
                else:
                    for _, row in top_plans.iterrows():
                        response_message += format_sales_data(row['PlanName'], row['total_sales'], row['total_revenue']) + "\n\n"

        except Exception as e:
            logging.error(f"Error while processing top sales plans: {e}")
            dispatcher.utter_message(text="An error occurred while processing your request. Please try again.")
            return []

        dispatcher.utter_message(text=response_message if response_message.strip() else "No sales data found for the specified criteria.")
        return []

class ActionLowestPlansSales(Action):

    def name(self) -> str:
        return "action_lowest_plans_sales"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        logging.info("Running ActionLowestPlansSales...")

        global LOADED_DATASET

        # Load the dataset if not already loaded
        if LOADED_DATASET is None:
            LOADED_DATASET = load_sales_data()
            if LOADED_DATASET is None:
                dispatcher.utter_message(text="Sales data could not be loaded. Please check the dataset source.")
                return []

        user_message = tracker.latest_message.get('text')
        df = LOADED_DATASET.copy()

        years = extract_years(user_message)
        months_extracted = extract_months(user_message)
        month_year_pairs = extract_month_year(user_message)
        response_message = ""

        # Helper function to format sales data
        def format_sales_data(planname, sales_count, total_revenue):
            return (f"Plan: {planname} \n "
                    f"Total Sales: {sales_count} \n "
                    f"Revenue: ${total_revenue:,.2f}")

        try:
            if month_year_pairs:
                month, year = month_year_pairs[0]
                response_message += f"📅 Lowest Sales in {months_reverse[month].capitalize()} {year}:\n\n"
                lowest_plans = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nsmallest(10, 'total_sales').reset_index()
                if lowest_plans.empty:
                    response_message += "No data available for the specified month and year.\n"
                else:
                    for _, row in lowest_plans.iterrows():
                        response_message += format_sales_data(row['PlanName'], row['total_sales'], row['total_revenue']) + "\n\n"

            elif years:
                for year in years:
                    response_message += f"📅 Lowest Sales in {year}:\n\n"
                    lowest_plans = df[df['PurchaseDate'].dt.year == year].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nsmallest(10, 'total_sales').reset_index()
                    if lowest_plans.empty:
                        response_message += f"No data available for the year {year}.\n"
                    else:
                        for _, row in lowest_plans.iterrows():
                            response_message += format_sales_data(row['PlanName'], row['total_sales'], row['total_revenue']) + "\n\n"

            elif months_extracted:
                current_year = datetime.now().year
                for month in months_extracted:
                    response_message += f"📅 Lowest Sales in {months_reverse[month].capitalize()} {current_year}:\n\n"
                    lowest_plans = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nsmallest(10, 'total_sales').reset_index()
                    if lowest_plans.empty:
                        response_message += f"No data available for {months_reverse[month]} {current_year}.\n"
                    else:
                        for _, row in lowest_plans.iterrows():
                            response_message += format_sales_data(row['PlanName'], row['total_sales'], row['total_revenue']) + "\n\n"

            else:
                response_message += "📊 Overall Lowest Sales Plans:\n\n"
                lowest_plans = df.groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nsmallest(10, 'total_sales').reset_index()
                if lowest_plans.empty:
                    response_message += "No data available overall.\n"
                else:
                    for _, row in lowest_plans.iterrows():
                        response_message += format_sales_data(row['PlanName'], row['total_sales'], row['total_revenue']) + "\n\n"

        except Exception as e:
            logging.error(f"Error while processing lowest sales plans: {e}")
            dispatcher.utter_message(text="An error occurred while processing your request. Please try again.")
            return []

        dispatcher.utter_message(text=response_message if response_message.strip() else "No sales data found for the specified criteria.")
        return []



        
########################################################################################################### top highest and lowest country sales #########################
class ActionTopHighestSalesByCountry(Action):
    def name(self) -> str:
        return "action_top_highest_sales_by_country"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        global LOADED_DATASET

        # Load the dataset if not already loaded
        if LOADED_DATASET is None:
            LOADED_DATASET = load_sales_data()
            if LOADED_DATASET is None:
                dispatcher.utter_message(text="Sales data could not be loaded. Please check the dataset source.")
                return []

        df = LOADED_DATASET.copy()
        user_message = tracker.latest_message.get('text')
        
        # Extract filters from the user's message
        years = extract_years(user_message)
        months_extracted = extract_months(user_message)
        month_year_pairs = extract_month_year(user_message)

        response_message = "📊 Top 10 Highest Sales Countries:\n\n"

        # Helper function to calculate top 10 highest sales by country
        def calculate_top_highest_sales_by_country(df, year=None, month=None):
            df_filtered = df.copy()

            try:
                if year:
                    df_filtered = df_filtered[df_filtered['PurchaseDate'].dt.year == year]
                if month:
                    df_filtered = df_filtered[df_filtered['PurchaseDate'].dt.month == month]

                # Group by country and calculate sales count and total revenue
                df_grouped = df_filtered.groupby('countryname').agg(
                    SalesCount=('SellingPrice', 'count'),
                    TotalRevenue=('SellingPrice', 'sum')
                ).sort_values('TotalRevenue', ascending=False).head(10).reset_index()

                return df_grouped
            except Exception as e:
                dispatcher.utter_message(text=f"Error processing sales data: {str(e)}")
                return pd.DataFrame()

        # If month and year are provided, show results for that specific month/year
        if month_year_pairs:
            for month, year in month_year_pairs:
                top_sales = calculate_top_highest_sales_by_country(df, year, month)
                if top_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for {month} {year}.")
                    continue
                response_message += f"🔍 Sales Overview for {month} {year}:\n"
                for index, row in top_sales.iterrows():
                    response_message += (
                        f"- {row['countryname']}: \n"
                        f"  Sales Count: {row['SalesCount']} \n"
                        f"  Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                response_message += "\n"

        # If only year is provided, show results for the entire year
        elif years:
            for year in years:
                top_sales = calculate_top_highest_sales_by_country(df, year)
                if top_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for {year}.")
                    continue
                response_message += f"🔍 Sales Overview for {year}:\n"
                for index, row in top_sales.iterrows():
                    response_message += (
                        f"- {row['countryname']}: \n"
                        f"  Sales Count: {row['SalesCount']} \n"
                        f"  Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                response_message += "\n"

        # If only month is provided, show results for that month in the current year
        elif months_extracted:
            current_year = datetime.now().year
            for month in months_extracted:
                top_sales = calculate_top_highest_sales_by_country(df, current_year, month)
                if top_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for {month} {current_year}.")
                    continue
                response_message += f"🔍 Sales Overview for {month} {current_year}:\n"
                for index, row in top_sales.iterrows():
                    response_message += (
                        f"- {row['countryname']}: \n"
                        f"  Sales Count: {row['SalesCount']} \n"
                        f"  Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                response_message += "\n"

        # If no filters, show overall top 10 highest sales by country
        else:
            top_sales = calculate_top_highest_sales_by_country(df)
            if top_sales.empty:
                dispatcher.utter_message(text="No sales data found.")
                return []
            for index, row in top_sales.iterrows():
                response_message += (
                    f"- {row['countryname']}: \n"
                    f"  Sales Count: {row['SalesCount']} \n"
                    f"  Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                )

        if not response_message.strip():
            dispatcher.utter_message(text="No sales data found for the specified criteria.")
            return []

        # Send the formatted message
        dispatcher.utter_message(text=response_message)
        return []


class ActionTopLowestSalesByCountry(Action):
    def name(self) -> str:
        return "action_top_lowest_sales_by_country"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        global LOADED_DATASET

        # Load the dataset if not already loaded
        if LOADED_DATASET is None:
            LOADED_DATASET = load_sales_data()  # Make sure this function is defined to load your data
            if LOADED_DATASET is None:
                dispatcher.utter_message(text="Sales data could not be loaded. Please check the dataset source.")
                return []

        df = LOADED_DATASET.copy()
        user_message = tracker.latest_message.get('text')
        
        # Extract filters from the user's message
        years = extract_years(user_message)
        months_extracted = extract_months(user_message)
        month_year_pairs = extract_month_year(user_message)

        response_message = "📉 Top 10 Lowest Sales Countries:\n\n"

        def calculate_top_lowest_sales_by_country(df: pd.DataFrame, year: int = None, month: str = None) -> pd.DataFrame:
            df_filtered = df.copy()

            try:
                if year:
                    df_filtered = df_filtered[df_filtered['PurchaseDate'].dt.year == year]
                if month:
                    df_filtered = df_filtered[df_filtered['PurchaseDate'].dt.month == month]

                # Group by country and calculate sales metrics
                df_grouped = df_filtered.groupby('countryname').agg(
                    SalesCount=('SellingPrice', 'count'),
                    TotalRevenue=('SellingPrice', 'sum')
                ).sort_values('TotalRevenue').head(10).reset_index()
                return df_grouped
            except Exception as e:
                dispatcher.utter_message(text=f"Error processing sales data: {str(e)}")
                return pd.DataFrame()

        if month_year_pairs:
            for month, year in month_year_pairs:
                lowest_sales = calculate_top_lowest_sales_by_country(df, year, month)
                if lowest_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for {month} {year}.")
                    continue
                response_message += f"🔍 Sales Overview for {month} {year}:\n"
                for index, row in lowest_sales.iterrows():
                    response_message += (
                        f"- {row['countryname']}: \n"
                        f"  Sales Count: {row['SalesCount']} \n"
                        f"  Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                response_message += "\n"

        elif years:
            for year in years:
                lowest_sales = calculate_top_lowest_sales_by_country(df, year)
                if lowest_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for {year}.")
                    continue
                response_message += f"🔍 Sales Overview for {year}:\n"
                for index, row in lowest_sales.iterrows():
                    response_message += (
                        f"- {row['countryname']}: \n"
                        f"  Sales Count: {row['SalesCount']} \n"
                        f"  Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                response_message += "\n"

        elif months_extracted:
            current_year = datetime.now().year
            for month in months_extracted:
                lowest_sales = calculate_top_lowest_sales_by_country(df, current_year, month)
                if lowest_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for {month} {current_year}.")
                    continue
                response_message += f"🔍 Sales Overview for {month} {current_year}:\n"
                for index, row in lowest_sales.iterrows():
                    response_message += (
                        f"- {row['countryname']}: \n"
                        f"  Sales Count: {row['SalesCount']} \n"
                        f"  Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                response_message += "\n"

        else:
            lowest_sales = calculate_top_lowest_sales_by_country(df)
            if lowest_sales.empty:
                dispatcher.utter_message(text="No sales data found.")
                return []
            for index, row in lowest_sales.iterrows():
                response_message += (
                    f"- {row['countryname']}: \n"
                    f"  Sales Count: {row['SalesCount']} \n"
                    f"  Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                )

        if not response_message.strip():
            dispatcher.utter_message(text="No sales data found for the specified criteria.")
            return []

        dispatcher.utter_message(text=response_message)
        return []




#################################################compare countries sales#######################################################################

class ActionCompareCountries(Action):
    def name(self) -> Text:
        return "action_compare_countries"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        global LOADED_DATASET

        # Load the dataset if not already loaded
        if LOADED_DATASET is None:
            LOADED_DATASET = load_sales_data()  # Ensure this function is defined to load your data
            if LOADED_DATASET is None:
                dispatcher.utter_message(text="Sales data could not be loaded. Please check the dataset source.")
                return []

        df = LOADED_DATASET.copy()
        user_message = tracker.latest_message.get('text')

        logging.info(f"User message: {user_message}")
        logging.info(f"Comparing country sales")

        # Extract countries, year, and month
        countries = extract_country(user_message, df['countryname'].dropna().unique().tolist())
        year = extract_years(user_message)  # Implement logic to extract year from message
        month = extract_months(user_message)  # Implement logic to extract month from message

        # Validate that two countries are provided for comparison
        if len(countries) != 2:
            dispatcher.utter_message(text="Please provide two countries for comparison.")
            return []

        country1, country2 = countries[0], countries[1]

        # Filter data by countries and time period
        try:
            if month and year:
                # Compare specific month and year
                df_country1 = filter_by_country_year_month(df, country=country1, year=year, month=month)
                df_country2 = filter_by_country_year_month(df, country=country2, year=year, month=month)
                comparison_type = f"{month}/{year}"
            elif year:
                # Compare whole year
                df_country1 = filter_by_country_year(df, country=country1, year=year)
                df_country2 = filter_by_country_year(df, country=country2, year=year)
                comparison_type = f"Year {year}"
            else:
                dispatcher.utter_message(text="Please specify a valid year or month and year for comparison.")
                return []

            # Check if filtered data is empty
            if df_country1.empty or df_country2.empty:
                dispatcher.utter_message(text=f"No sales data found for {country1} or {country2} for {comparison_type}.")
                return []

            # Generate comparison results
            comparison_result = self.compare_sales(df_country1, df_country2, country1, country2, comparison_type)
            dispatcher.utter_message(text=f"Comparison of sales between {country1} and {country2} for {comparison_type}:\n\n{comparison_result}")
        
        except Exception as e:
            logging.error(f"Error during comparison: {e}")
            dispatcher.utter_message(text="An error occurred while comparing sales. Please try again later.")
            return []

        return []

    def compare_sales(self, df_country1, df_country2, country1, country2, comparison_type):
        result = f"Sales Comparison for {comparison_type}:\n\n"

        try:
            # Total sales comparison
            total_sales_amount_country1 = df_country1['SellingPrice'].sum()
            total_sales_amount_country2 = df_country2['SellingPrice'].sum()
            total_sales_count_country1 = df_country1['SellingPrice'].count()
            total_sales_count_country2 = df_country2['SellingPrice'].count()

            result += f"Total Sales in {country1}: {total_sales_count_country1} sales, generating ${total_sales_amount_country1:,.2f}\n"
            result += f"Total Sales in {country2}: {total_sales_count_country2} sales, generating ${total_sales_amount_country2:,.2f}\n\n"

            # Calculate sales count and amount differences with descriptive messages
            sales_count_difference = abs(total_sales_count_country1 - total_sales_count_country2)
            sales_amount_difference = abs(total_sales_amount_country1 - total_sales_amount_country2)

            if total_sales_amount_country1 > total_sales_amount_country2:
                result += f"{country1} had ${sales_amount_difference:,.2f} more revenue and {sales_count_difference} more sales than {country2}.\n\n"
            elif total_sales_amount_country1 < total_sales_amount_country2:
                result += f"{country2} had ${sales_amount_difference:,.2f} more revenue and {sales_count_difference} more sales than {country1}.\n\n"
            else:
                result += f"Sales revenue or counts were equal between {country1} and {country2}.\n\n"

            # Generate top and least plans and sources
            result += f"Top 5 plans by sales in {country1}:\n"
            result += self.get_top_plans(df_country1)
            result += f"\nTop 5 plans by sales in {country2}:\n"
            result += self.get_top_plans(df_country2)

            result += f"\nLeast 5 plans by sales in {country1}:\n"
            result += self.get_least_plans(df_country1)
            result += f"\nLeast 5 plans by sales in {country2}:\n"
            result += self.get_least_plans(df_country2)

            result += f"\nTop 5 sources by sales in {country1}:\n"
            result += self.get_top_sources(df_country1)
            result += f"\nTop 5 sources by sales in {country2}:\n"
            result += self.get_top_sources(df_country2)

            result += f"\nTop 3 payment gateways by sales in {country1}:\n"
            result += self.get_top_payment_gateways(df_country1)
            result += f"\nTop 3 payment gateways by sales in {country2}:\n"
            result += self.get_top_payment_gateways(df_country2)

            result += f"\nTop 6 refsite by sales in {country1}:\n"
            result += self.get_top_refsites(df_country1)
            result += f"\nTop 6 refsite by sales in {country2}:\n"
            result += self.get_top_refsites(df_country2)

        except Exception as e:
            logging.error(f"Error in sales comparison: {e}")
            return "An error occurred while generating the sales comparison."

        return result

    def get_top_plans(self, df):
        try:
            plans_counts = df.groupby('PlanName').agg(total_sales=('SellingPrice', 'sum'), sales_count=('SellingPrice', 'count')).nlargest(5, 'total_sales')
            return "\n".join([f"{plan}: ${sales:,.2f} with {count} sales" for plan, (sales, count) in plans_counts.iterrows()])
        except Exception as e:
            logging.error(f"Error fetching top plans: {e}")
            return "Could not retrieve top plans."

    def get_least_plans(self, df):
        try:
            plans_counts = df.groupby('PlanName').agg(total_sales=('SellingPrice', 'sum'), sales_count=('SellingPrice', 'count')).nsmallest(5, 'total_sales')
            return "\n".join([f"{plan}: ${sales:,.2f} with {count} sales" for plan, (sales, count) in plans_counts.iterrows()])
        except Exception as e:
            logging.error(f"Error fetching least plans: {e}")
            return "Could not retrieve least plans."

    def get_top_sources(self, df):
        try:
            source_counts = df.groupby('source').agg(total_sales=('SellingPrice', 'sum'), sales_count=('SellingPrice', 'count')).nlargest(5, 'total_sales')
            return "\n".join([f"{source}: ${sales:,.2f} with {count} sales" for source, (sales, count) in source_counts.iterrows()])
        except Exception as e:
            logging.error(f"Error fetching top sources: {e}")
            return "Could not retrieve top sources."

    def get_top_payment_gateways(self, df):
        try:
            gateway_counts = df.groupby('payment_gateway').agg(total_sales=('SellingPrice', 'sum'), sales_count=('SellingPrice', 'count')).nlargest(3, 'total_sales')
            return "\n".join([f"{gateway}: ${sales:,.2f} with {count} transactions" for gateway, (sales, count) in gateway_counts.iterrows()])
        except Exception as e:
            logging.error(f"Error fetching top payment gateways: {e}")
            return "Could not retrieve top payment gateways."

    def get_top_refsites(self, df):
        try:
            refsite_counts = df.groupby('Refsite').agg(total_sales=('SellingPrice', 'sum'), sales_count=('SellingPrice', 'count')).nlargest(6, 'total_sales')
            return "\n".join([f"{refsite}: ${sales:,.2f} with {count} transactions" for refsite, (sales, count) in refsite_counts.iterrows()])
        except Exception as e:
            logging.error(f"Error fetching top refsites: {e}")
            return "Could not retrieve top refsites."




##############################################################################################################most and least buying sales plan for each country#########################

class ActionMostAndLeastSoldPlansForCountry(Action):
    def name(self) -> str:
        return "action_most_and_least_sold_plans_for_country"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        global LOADED_DATASET
        
        # Load the dataset if not already loaded
        if LOADED_DATASET is None:
            LOADED_DATASET = load_sales_data()
            if LOADED_DATASET is None:
                dispatcher.utter_message(text="Sales data could not be loaded. Please check the dataset source.")
                return []

        df = LOADED_DATASET.copy()
        user_message = tracker.latest_message.get('text')

        # Extract filters from the user's message
        country = tracker.get_slot("country")
        years = extract_years(user_message)
        months_extracted = extract_months(user_message)
        month_year_pairs = extract_month_year(user_message)

        response_message = f"📊 Sales Overview for {country}:\n\n"

        # Helper function to calculate most and least sold plans
        def calculate_sales_by_country(df, country, year=None, month=None):
            try:
                df_filtered = df.copy()
                
                # Filter by country
                df_filtered = df_filtered[df_filtered['countryname'].str.lower() == country.lower()]
                if year:
                    df_filtered = df_filtered[df_filtered['PurchaseDate'].dt.year == year]
                if month:
                    df_filtered = df_filtered[df_filtered['PurchaseDate'].dt.month == month]

                # Group by plan name and calculate sales count and total revenue
                df_grouped = df_filtered.groupby('PlanName').agg(
                    SalesCount=('SellingPrice', 'count'),
                    TotalRevenue=('SellingPrice', 'sum')
                ).reset_index()

                return df_grouped

            except Exception as e:
                dispatcher.utter_message(text=f"An error occurred while calculating sales data: {str(e)}")
                return pd.DataFrame()  # Return an empty DataFrame on error

        # If month and year are provided, show results for that specific month/year
        if month_year_pairs:
            for month, year in month_year_pairs:
                top_sales = calculate_sales_by_country(df, country, year, month)
                if top_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for {month} {year} in {country}.")
                    return []
                
                # Most sold plans
                most_sold = top_sales.nlargest(5, 'SalesCount')
                response_message += f"🔍 Sales Overview for {month} {year} in {country}:\n"
                for index, row in most_sold.iterrows():
                    response_message += (
                        f"  - {row['PlanName']}\n"
                        f"    Sales Count: {row['SalesCount']}\n"
                        f"    Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                
                # Least sold plans
                least_sold = top_sales.nsmallest(5, 'SalesCount')
                response_message += f"\n Least Sold Plans in {country}:\n"
                for index, row in least_sold.iterrows():
                    response_message += (
                        f"  - {row['PlanName']}\n"
                        f"    Sales Count: {row['SalesCount']}\n"
                        f"    Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                response_message += "\n"

        # If only year is provided, show results for the entire year
        elif years:
            for year in years:
                top_sales = calculate_sales_by_country(df, country, year)
                if top_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for {year} in {country}.")
                    return []
                
                # Most sold plans
                most_sold = top_sales.nlargest(5, 'SalesCount')
                response_message += f"🔍 Sales Overview for {year} in {country}:\n"
                for index, row in most_sold.iterrows():
                    response_message += (
                        f"  - {row['PlanName']}\n"
                        f"    Sales Count: {row['SalesCount']}\n"
                        f"    Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                
                # Least sold plans
                least_sold = top_sales.nsmallest(5, 'SalesCount')
                response_message += f"\n Least Sold Plans in {country}:\n"
                for index, row in least_sold.iterrows():
                    response_message += (
                        f"  - {row['PlanName']}\n"
                        f"    Sales Count: {row['SalesCount']}\n"
                        f"    Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                response_message += "\n"

        # If only month is provided, show results for that month in the current year
        elif months_extracted:
            current_year = datetime.now().year
            for month in months_extracted:
                top_sales = calculate_sales_by_country(df, country, current_year, month)
                if top_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for {month} {current_year} in {country}.")
                    return []
                
                # Most sold plans
                most_sold = top_sales.nlargest(5, 'SalesCount')
                response_message += f"🔍 Sales Overview for {month} {current_year} in {country}:\n"
                for index, row in most_sold.iterrows():
                    response_message += (
                        f"  - {row['PlanName']}\n"
                        f"    Sales Count: {row['SalesCount']}\n"
                        f"    Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                
                # Least sold plans
                least_sold = top_sales.nsmallest(5, 'SalesCount')
                response_message += f"\n Least Sold Plans in {country}:\n"
                for index, row in least_sold.iterrows():
                    response_message += (
                        f"  - {row['PlanName']}\n"
                        f"    Sales Count: {row['SalesCount']}\n"
                        f"    Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                response_message += "\n"

        # If no filters, show overall top 10 highest sold plans by country
        else:
            top_sales = calculate_sales_by_country(df, country)
            if top_sales.empty:
                dispatcher.utter_message(text=f"No sales data found for {country}.")
                return []
            
            # Most sold plans
            most_sold = top_sales.nlargest(5, 'SalesCount')
            response_message += f"🔍 Overall Most Sold Plans in {country}:\n"
            for index, row in most_sold.iterrows():
                response_message += (
                    f"  - {row['PlanName']}\n"
                    f"    Sales Count: {row['SalesCount']}\n"
                    f"    Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                )
            
            # Least sold plans
            least_sold = top_sales.nsmallest(5, 'SalesCount')
            response_message += f"\n 🔍 Overall Least Sold Plans in {country}:\n"
            for index, row in least_sold.iterrows():
                response_message += (
                    f"  - {row['PlanName']}\n"
                    f"    Sales Count: {row['SalesCount']}\n"
                    f"    Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                )

        if not response_message.strip():
            dispatcher.utter_message(text="No sales data found for the specified criteria.")
            return []

        # Send the formatted message
        dispatcher.utter_message(text=response_message)
        return []





#########################################################################################################################calculate refsite,source and payment gateway sales########

class ActionSalesBySourcePaymentGatewayRefsite(Action):

    def name(self) -> str:
        return "action_sales_by_source_payment_gateway_refsite"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        logging.info("Running ActionSalesBySourcePaymentGatewayRefsite...")
        
        global LOADED_DATASET

        # Load the dataset if not already loaded
        if LOADED_DATASET is None:
            LOADED_DATASET = load_sales_data()
            if LOADED_DATASET is None:
                dispatcher.utter_message(text="Sales data could not be loaded. Please check the dataset source.")
                return []

        user_message = tracker.latest_message.get('text')
        df = LOADED_DATASET.copy()
        
        # Check if required columns are present
        required_columns = ['PurchaseDate', 'SellingPrice', 'source', 'payment_gateway', 'Refsite']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            dispatcher.utter_message(text=f"Data is missing the following required columns: {', '.join(missing_columns)}")
            return []

        source = next(tracker.get_latest_entity_values('source'), None)
        payment_gateway = next(tracker.get_latest_entity_values('payment_gateway'), None)
        refsite = next(tracker.get_latest_entity_values('refsite'), None)

        # Normalize source column by converting to lowercase
        df['source'] = df['source'].str.lower()

        # Extract time conditions from the user message
        years = extract_years(user_message)
        months_extracted = extract_months(user_message)
        month_year_pairs = extract_month_year(user_message)

        response_message = ""

        # Helper function to format sales data
        def format_sales_data(category, name, total_revenue, sales_count):
            return (f"{category.capitalize()}: {name}\n"
                    f"Total Revenue: ${total_revenue:,.2f}\n"
                    f"Sales Count: {sales_count}\n\n")

        # Generate sales data based on time conditions
        try:
            if month_year_pairs:
                response_message += f"📅 Sales Overview by Source, Payment Gateway, and Refsite for {months_reverse[month_year_pairs[0][0]].capitalize()} {month_year_pairs[0][1]}:\n\n"
                for month, year in month_year_pairs:
                    filtered_df = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)]
                    if filtered_df.empty:
                        response_message += f"No sales data found for {months_reverse[month]} {year}.\n"
                    else:
                        response_message += self.process_sales_data(filtered_df, format_sales_data)

            elif years:
                response_message += f"📅 Sales Overview by Source, Payment Gateway, and Refsite for {years[0]}:\n\n"
                for year in years:
                    filtered_df = df[df['PurchaseDate'].dt.year == year]
                    if filtered_df.empty:
                        response_message += f"No sales data found for {year}.\n"
                    else:
                        response_message += self.process_sales_data(filtered_df, format_sales_data)

            elif months_extracted:
                current_year = datetime.now().year
                response_message += f"📅 Sales Overview by Source, Payment Gateway, and Refsite for {months_reverse[months_extracted[0]].capitalize()} {current_year}:\n\n"
                for month in months_extracted:
                    filtered_df = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)]
                    if filtered_df.empty:
                        response_message += f"No sales data found for {months_reverse[month]} {current_year}.\n"
                    else:
                        response_message += self.process_sales_data(filtered_df, format_sales_data)

            else:
                response_message += "📊 Overall Sales Overview by Source, Payment Gateway, and Refsite:\n\n"
                response_message += self.process_sales_data(df, format_sales_data)

            # If no data is available after processing, inform the user
            if not response_message.strip():
                dispatcher.utter_message(text="No sales data found for the specified criteria.")
                return []

            # Send the formatted message
            dispatcher.utter_message(text=response_message)

        except Exception as e:
            logging.error(f"An error occurred while processing sales data: {str(e)}")
            dispatcher.utter_message(text="An error occurred while processing the sales data. Please try again later.")

        return []

    def process_sales_data(self, filtered_df, format_sales_data):
        """Processes sales data to get highest and lowest sales for each category."""
        response = ""

        # Process sources
        if 'source' in filtered_df.columns:
            response += "Sales Overview by Source:\n"
            response += self.get_top_n_sales(filtered_df, 'source', format_sales_data)

        # Process payment gateways
        if 'payment_gateway' in filtered_df.columns:
            response += "\nSales Overview by Payment Gateway:\n"
            response += self.get_top_n_sales(filtered_df, 'payment_gateway', format_sales_data)

        # Process refsites
        if 'Refsite' in filtered_df.columns:
            response += "\nSales Overview by Refsite:\n"
            response += self.get_top_n_sales(filtered_df, 'Refsite', format_sales_data)

        return response

    def get_top_n_sales(self, df, column, format_sales_data, n=5):
        """Helper function to calculate top N highest and lowest sales."""
        response = ""

        # Aggregate data by column
        sales_summary = df.groupby(column)['SellingPrice'].agg(['sum', 'count']).reset_index()
        sales_summary.columns = [column, 'TotalRevenue', 'SalesCount']

        # Get top N highest sales
        top_sales = sales_summary.nlargest(n, 'SalesCount')
        response += f"Top {n} Highest Sales by {column.capitalize()}:\n"
        for index, row in top_sales.iterrows():
            response += format_sales_data(column, row[column], row['TotalRevenue'], row['SalesCount'])

        # Get top N lowest sales
        lowest_sales = sales_summary.nsmallest(n, 'SalesCount')
        response += f"\nTop {n} Lowest Sales by {column.capitalize()}:\n"
        for index, row in lowest_sales.iterrows():
            response += format_sales_data(column, row[column], row['TotalRevenue'], row['SalesCount'])

        return response

################################################################################################################calculate total sales for each month and each year, sales growth ####################################

class ActionCalculateSalesMetricsAndGrowth(Action):

    def name(self) -> str:
        return "action_calculate_sales_metrics_and_growth"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logging.info("Running ActionCalculateSalesMetrics...")

        global LOADED_DATASET

        # Load the dataset if not already loaded
        if LOADED_DATASET is None:
            LOADED_DATASET = load_sales_data()
            if LOADED_DATASET is None:
                dispatcher.utter_message(text="Sales data could not be loaded. Please check the dataset source.")
                return []

        try:
            df = LOADED_DATASET.copy()
            # Check if the necessary columns exist
            if 'PurchaseDate' not in df.columns or 'SellingPrice' not in df.columns:
                dispatcher.utter_message(text="Required columns 'PurchaseDate' or 'SellingPrice' are missing from the dataset.")
                return []

            df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
            df = df.dropna(subset=['PurchaseDate', 'SellingPrice'])  # Drop rows with invalid dates or sales data

            if df.empty:
                dispatcher.utter_message(text="No valid sales data available for analysis.")
                return []

            df['MonthYear'] = df['PurchaseDate'].dt.to_period('M')
            df['Year'] = df['PurchaseDate'].dt.year

            # Calculate monthly sales totals and counts
            monthly_data = df.groupby("MonthYear").agg(
                TotalSales=('SellingPrice', 'sum'),
                SalesCount=('SellingPrice', 'count')
            ).reset_index()

            # Prepare the response message
            response_message = "📊 Sales Metrics Overview:\n\n"

            # Sales Overview
            response_message += " 📅 Monthly Sales Summary:\n"
            if monthly_data.empty:
                response_message += "No monthly sales data available.\n\n"
            else:
                for index, row in monthly_data.iterrows():
                    response_message += (
                        f" {row['MonthYear']} \n"
                        f" Total Sales: ${row['TotalSales']:,.2f}\n"
                        f" Sales Count: {row['SalesCount']}\n\n"
                    )

            # Prepare yearly summary
            yearly_summary = df.groupby('Year').agg(
                TotalSales=('SellingPrice', 'sum'),
                SalesCount=('SellingPrice', 'count')
            ).reset_index()

            # Add yearly data to the response
            response_message += " 📅 Yearly Sales Summary:\n"
            if yearly_summary.empty:
                response_message += "No yearly sales data available.\n\n"
            else:
                for index, row in yearly_summary.iterrows():
                    response_message += (
                        f"{row['Year']}\n"
                        f"Total Sales: ${row['TotalSales']:,.2f} \n"
                        f"Sales Count: {row['SalesCount']}\n\n"
                    )

            # Calculate growth percentages
            monthly_growth = monthly_data['SalesCount'].pct_change() * 100
            yearly_growth = yearly_summary['SalesCount'].pct_change() * 100

            # Growth Overview
            response_message += "Growth Overview:\n\n"
            
            # Monthly Growth
            response_message += " 📈 Monthly Growth Percentage:\n"
            for index in range(1, len(monthly_data)):
                growth = monthly_growth[index]
                response_message += (
                    f"From {monthly_data['MonthYear'][index-1]} to {monthly_data['MonthYear'][index]}  \n"
                    f"Growth: {growth:.2f}%\n\n"
                )

            # Yearly Growth
            response_message += "📈 Yearly Growth Percentage:\n"
            for index in range(1, len(yearly_summary)):
                growth = yearly_growth[index]
                response_message += (
                    f"From {yearly_summary['Year'][index-1]} to {yearly_summary['Year'][index]}  \n"
                    f"Growth: {growth:.2f}%\n\n"
                )

            # Send the formatted message
            dispatcher.utter_message(text=response_message)

        except Exception as e:
            logging.error(f"Error while calculating sales metrics: {str(e)}")
            dispatcher.utter_message(text="An error occurred while calculating sales metrics. Please try again later.")

        return []



