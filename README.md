# Recommendation System Project

## Overview
This project implements a recommendation system using hybrid approaches (content-based and collaborative filtering) to suggest videos to users based on their interactions, category preferences, and mood.

## Requirements
- Python 3.x
- FastAPI
- Pandas
- scikit-learn
- Requests
- pytest

## Setup Instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/recommendation-system-project.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the FastAPI server:
    ```bash
    uvicorn src.api:app --reload
    ```

4. Access the API endpoints at:
    - `http://localhost:8000/feed?username=your_username&category_id=category_id_user_want_to_see&mood=user_current_mood`
    - `http://localhost:8000/feed?username=your_username&category_id=category_id_user_want_to_see`
    - `http://localhost:8000/feed?username=your_username`

## Testing
1. To run the tests, use:
    ```bash
    pytest
    ```

## Video Presentation
- Watch the video walkthrough [here](URL to video).
