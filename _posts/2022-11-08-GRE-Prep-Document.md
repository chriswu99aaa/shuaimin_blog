The purpose of this website is for education.

The targeted audiences are students who want to take GRE, and this website will provide personalized guidance and quantified summury to their exam preparation.

Audience Age: 23-28

Business Requirement:

1. Users can take GRE preparation question on the web
2. Users can login by username and password
3. Users can make payment to get better service
4. Users can have free trials
5. Users can see a quantified summury of their exam performance. For example, they can see their performance under certain category of question, and changes in performance over the past weeks or month
6. Users can see explanation for questions: *Question*: can we use XAI in NLP to find the reason to determine the answer
7. Users authentication


## Backend Consideration:

Data Required:

* GRE Questions
  * question should have related category tag
* DB: which DBMS do I use?
  * User: email, id, password, 
  * Categories of question: verbal, logical thinking, etc
  * performance in each category
* contact information: 
  * a commercial email only for this information
* image: **Pending**

|Name|Attributes|
|----|----------|
|User|id(primary key),email, username, password, Questions done, membership status |
|Performance|Serial No. of each in take (id primary key), the amount correctly answered in each in take for each category, User id(who took this exam) | how many question should a in take consists? 20 for testing purpose
|Question|id(primary key),catetories of this question, answer of this question|
|Answer|id(primary key), Question id|
|Category|list of categories|
|Membership|id, membership level|
|||

## Design Storyboard

1. Create a template for webiste
2. Design website hierachy

Design should be consistency, light color background, and dark color for text with sufficient white space

Use an simulator to see how webiste look like on various devices

## Software Development Life Cycle

1. Analysis
2. Deisgn
3. Code
4. Testing
5. Deployment

## Performance Testing

1. Maximum User Load
2. response time,
3. question loading time
4. data loading time

How can we use cache in a browser

1. Analyse the requirement
2. UI Design
3. Website hierachy





















