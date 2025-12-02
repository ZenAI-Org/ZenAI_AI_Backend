from datetime import datetime, timedelta 
import re 

def parse_relative_date(date_string:str) -> str: 
    """ 
        Convert natural language dates to YYYY-MM-DD format 
        Examples; Tomorrow, Friday , Next MOnday , etc """ 
    if not date_string: 
        return None 
    
    date_string = date_string.lower().strip() 
    today = datetime.now() 
    
    # If it is already in YYYY-MM-SS format 
    if re.match(r'\d{4}-\d{2}-\d{2}', date_string):
        return date_string 

    # For Yesterday 
    if date_string == 'yesterday': 
        return (today - timedelta(days = 1)).strftime('%Y-%m-%d') 
    # For tomorrow 
    if date_string == 'tomorrow': 
        return (today + timedelta(days = 1)).strftime('%Y-%m-%d') 
    
    # For today 
    if date_string == 'today': 
        return today.strftime('%Y-%m-%d') 
    
    # Handles day names { Monday , Tuesday etc} 
    days = { 
            'monday': 0, 
            'tuesday': 1, 
            'wednesday': 2, 
            'thursday': 3, 
            'friday': 4, 
            'saturday': 5, 
            'sunday': 6 
        } 
    for day_name, day_num in days.items(): 
        if day_name in date_string: 
            current_day = today.weekday() 
            days_ahead = (day_num - current_day ) % 7 
            
            # Handle last Monday ( pa)s) 
            if "last" in date_string or "past" in date_string: 
                if days_ahead == 0 : 
                    days_ahead = -7 
                else: 
                    days_ahead = -(7 - days_ahead) 
                target_date = today + timedelta(days = days_ahead) 
                return target_date.strftime('%Y-%m-%d')
            # If it's same day , then we go for the next week 
            if days_ahead == 0: 
                days_ahead = 7 
            
            # Handles "next" keyword : next Monday vs just MOnday 
            if "next" in date_string:
                days_ahead += 7 
            
            target_date = today + timedelta(days = days_ahead) 
            return target_date.strftime('%Y-%m-%d') 
    
    # Handle in "X" days 
    match = re.search(r'in (\d+) days?', date_string) 
    if match:
        days = int(match.group(1)) 
        return (today + timedelta(days = days)).strftime('%Y-%m-%d') 
    
    # Handle "X' days age" 
    match = re.search(r'(\d+) days? ago', date_string) 
    if match: 
        days = int(match.group(1)) 
        return (today - timedelta(days = days)).strftime('%Y-%m-%d') 
    
    # Handles next week 
    if "next_week" in date_string:
        return (today + timedelta(days = 7)).strftime('%Y-%m-%d') 
    
    # Handle last week 
    if "last week" in date_string: 
        return (today - timedelta(days = 7)).strftime('%Y-%m-%d') 
    
    # If can't parse 
    print(f"[WARNING] Could not parse the date: '{date_string }") 
    return None 