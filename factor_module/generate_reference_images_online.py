import os
import time
from icrawler.builtin import GoogleImageCrawler

# ========== CONFIG ==========
OUTPUT_DIR = "data/reference_images_raw"
MAX_IDENTITIES = 500

# ========== SEED LIST (Famous Public Figures) ==========
IDENTITIES = [
    # Original 200 (already included in your list)
    "Barack Obama", "Donald Trump", "Elon Musk", "Emma Watson", "Tom Cruise",
    "Taylor Swift", "Keanu Reeves", "Cristiano Ronaldo", "Narendra Modi",
    "Beyonce", "Angelina Jolie", "Leonardo DiCaprio", "Greta Thunberg",
    "Shahrukh Khan", "Amitabh Bachchan", "Mark Zuckerberg", "Joe Biden",
    "Emma Stone", "Priyanka Chopra", "Dwayne Johnson", "Brad Pitt",
    "Virat Kohli", "Selena Gomez", "Bill Gates", "Zelensky",
    "Gal Gadot", "Kylie Jenner", "Hrithik Roshan", "Kendall Jenner", "Ranveer Singh",
    "Robert Downey Jr", "Chris Hemsworth", "Chris Evans", "Scarlett Johansson", "Natalie Portman",
    "Zendaya", "Jason Momoa", "Tom Holland", "Millie Bobby Brown", "Noah Schnapp",
    "Finn Wolfhard", "Sadie Sink", "Jennifer Lawrence", "Johnny Depp", "Will Smith",
    "Matthew McConaughey", "Ryan Reynolds", "Hugh Jackman", "Ben Affleck", "Matt Damon",
    "Anne Hathaway", "Morgan Freeman", "Meryl Streep", "Daniel Craig", "Henry Cavill",
    "Margot Robbie", "Christian Bale", "Cillian Murphy", "Florence Pugh", "Tobey Maguire",
    "Andrew Garfield", "Emma Roberts", "Kristen Stewart", "Robert Pattinson", "Galilea Montijo",
    "Salma Hayek", "Eva Longoria", "Sofia Vergara", "Eiza Gonzalez", "Luis Miguel",
    "Maluma", "Bad Bunny", "Karol G", "Rosalia", "Shakira",
    "Camila Cabello", "Anitta", "Jennifer Lopez", "Dua Lipa", "Ariana Grande",
    "Lady Gaga", "Doja Cat", "Nicki Minaj", "Megan Thee Stallion", "Iggy Azalea",
    "Post Malone", "Ed Sheeran", "Justin Bieber", "Drake", "The Weeknd",
    "Billie Eilish", "Shawn Mendes", "Zayn Malik", "Harry Styles", "Louis Tomlinson",
    "Liam Payne", "Niall Horan", "Camila Mendes", "Lili Reinhart", "Cole Sprouse",
    "KJ Apa", "Madelaine Petsch", "Vanessa Hudgens", "Ashley Tisdale", "Brenda Song",
    "Hilary Duff", "Demi Lovato", "Miley Cyrus", "Nick Jonas", "Joe Jonas",
    "Kevin Jonas", "Sophie Turner", "Maisie Williams", "Emilia Clarke", "Kit Harington",
    "Peter Dinklage", "Lena Headey", "Nathalie Emmanuel", "Isaac Hempstead Wright", "Millie Gibson",
    "Olivia Rodrigo", "Sabrina Carpenter", "Charli D'Amelio", "Dixie D'Amelio", "Addison Rae",
    "Avani Gregg", "Bryce Hall", "Josh Richards", "Griffin Johnson", "Noah Beck",
    "Zach King", "Khaby Lame", "Bella Poarch", "Nessa Barrett", "Chase Hudson",
    "Jaden Hossler", "Tana Mongeau", "David Dobrik", "Liza Koshy", "Emma Chamberlain",
    "James Charles", "Jeffree Star", "Bretman Rock", "NikkieTutorials", "MrBeast",
    "Markiplier", "PewDiePie", "Jacksepticeye", "DanTDM", "Dream",
    "GeorgeNotFound", "Sapnap", "Karl Jacobs", "TommyInnit", "Technoblade",
    "Wilbur Soot", "Tubbo", "Ranboo", "Philza", "Corpse Husband",
    "Pokimane", "Valkyrae", "Sykkuno", "xQc", "Shroud",
    "Ninja", "Tfue", "Myth", "TimTheTatman", "Dr Disrespect",
    "Seth Everman", "Michael Reeves", "William Osman", "Domics", "Jaiden Animations",
    "TheOdd1sOut", "Kurzgesagt", "Veritasium", "Mark Rober", "Tom Scott",
    "Colleen Ballinger", "Miranda Sings", "Gabbie Hanna", "Trisha Paytas", "Shane Dawson",
    "Ryland Adams", "Garrett Watts", "Morgan Adams", "Andrew Siwicki", "Anthony Padilla",
    "Ian Hecox", "Courtney Miller", "Olivia Sui", "Noah Grossman", "Keith Leak Jr",
    "Shayne Topp", "Damien Haas", "Kim Kardashian", "Kanye West", "Pete Davidson",
    "Travis Scott", "Kourtney Kardashian", "Khloe Kardashian", "Rob Kardashian", "Blake Lively",
    "Ryan Gosling", "Rachel McAdams", "Eva Mendes", "Jessica Alba", "Mila Kunis",
    "Ashton Kutcher", "Channing Tatum", "Jenna Dewan", "Zac Efron", "Vanessa Kirby",
    "Eddie Redmayne", "Emma Thompson", "Helen Mirren", "Idris Elba", "Naomi Scott",
    "Dev Patel", "Freida Pinto", "Riz Ahmed", "Mindy Kaling", "Kal Penn",
    "Hasan Minhaj", "Nora Fatehi", "Tiger Shroff", "Kriti Sanon", "Kangana Ranaut",
    "Alia Bhatt", "Deepika Padukone", "Ranbir Kapoor", "Vicky Kaushal", "Taapsee Pannu",
    "Ananya Panday", "Janhvi Kapoor", "Sara Ali Khan", "Parineeti Chopra", "Rajkummar Rao",
    "Ayushmann Khurrana", "Pankaj Tripathi", "Nawazuddin Siddiqui", "Manoj Bajpayee", "Irrfan Khan",
    "Farhan Akhtar", "Arjun Kapoor", "John Abraham", "Saif Ali Khan", "Suniel Shetty",
    "Akshay Kumar", "Sanjay Dutt", "Jackie Shroff", "Anil Kapoor",

    # NEW ADDITIONS (300 more to reach 500)
    "Joe Rogan", "Lex Fridman", "Neil deGrasse Tyson", "Brian Cox", "Michio Kaku",
    "Sundar Pichai", "Satya Nadella", "Susan Wojcicki", "Reed Hastings", "Jeff Bezos",
    "Larry Page", "Sergey Brin", "Steve Jobs", "Tim Cook", "Linus Torvalds",
    "Jordan Peterson", "Ben Shapiro", "Andrew Tate", "Greta Gerwig", "Timothee Chalamet",
    "Aubrey Plaza", "Pedro Pascal", "Oscar Isaac", "Paul Rudd", "Adam Sandler",
    "Ben Stiller", "Jonah Hill", "Seth Rogen", "Zoe Kravitz", "Chloe Grace Moretz",
    "Dylan O'Brien", "Tyler Posey", "Victoria Justice", "Elizabeth Olsen", "Aubrey Anderson-Emmons",
    "David Schwimmer", "Matthew Perry", "Jennifer Aniston", "Courteney Cox", "Lisa Kudrow",
    "Matt LeBlanc", "Kaley Cuoco", "Jim Parsons", "Johnny Galecki", "Mayim Bialik",
    "Simon Helberg", "Kunal Nayyar", "Maitreyi Ramakrishnan", "Darren Barnet", "Ramona Young",
    "Poorna Jagannathan", "Richa Moorjani", "Sendhil Ramamurthy", "Ashley Greene", "Nikki Reed",
    "Ian Somerhalder", "Paul Wesley", "Nina Dobrev", "Candice King", "Kat Graham",
    "Phoebe Tonkin", "Daniel Gillies", "Claire Holt", "Joseph Morgan", "Tyler Blackburn",
    "Troian Bellisario", "Shay Mitchell", "Lucy Hale", "Ashley Benson", "Sasha Pieterse",
    "Jason Derulo", "Ne-Yo", "T-Pain", "Akon", "Flo Rida",
    "LMFAO", "Far East Movement", "OneRepublic", "Maroon 5", "Adam Levine",
    "Sia", "Halsey", "Bebe Rexha", "Tove Lo", "Ava Max",
    "Rita Ora", "Jessie J", "Clean Bandit", "Alan Walker", "Kygo",
    "Marshmello", "Skrillex", "Steve Aoki", "Zedd", "Martin Garrix",
    "Armin van Buuren", "David Guetta", "Hardwell", "Avicii", "Tiesto",
    "Benny Blanco", "Charlie Puth", "Shane Watson", "Glenn Maxwell", "AB de Villiers",
    "Kane Williamson", "Steve Smith", "David Warner", "Ben Stokes", "Joe Root",
    "Jasprit Bumrah", "MS Dhoni", "Rohit Sharma", "Yuvraj Singh", "Hardik Pandya",
    "KL Rahul", "Ravindra Jadeja", "Shubman Gill", "Suryakumar Yadav", "Rishabh Pant",
    "Dinesh Karthik", "Sanju Samson", "Kuldeep Yadav", "Axar Patel", "Washington Sundar",
    "Yuzvendra Chahal", "Mohammed Siraj", "Harshal Patel", "Umran Malik", "Shikhar Dhawan",
    "Imran Khan", "Wasim Akram", "Babar Azam", "Shaheen Afridi", "Mohammad Rizwan",
    "Tamim Iqbal", "Mashrafe Mortaza", "Shakib Al Hasan", "Mushfiqur Rahim", "Angelo Mathews",
    "Kumar Sangakkara", "Mahela Jayawardene", "Lasith Malinga", "Chris Gayle", "Andre Russell",
    "Sunil Narine", "Dwayne Bravo", "Kieron Pollard", "Jason Holder", "Rashid Khan",
    "Mohammad Nabi", "Mujeeb Ur Rahman", "Trent Boult", "Mitchell Starc", "Pat Cummins",
    "Josh Hazlewood", "Jofra Archer", "Mark Wood", "Sam Curran", "Alex Hales",
    "Jos Buttler", "Moeen Ali", "Chris Woakes", "Adil Rashid", "Eoin Morgan"
]


IDENTITIES = IDENTITIES[:MAX_IDENTITIES]

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"🧠 Targeting {len(IDENTITIES)} unique identities with 1 image each...")

# ========== DOWNLOAD ==========
success_count = 0
fail_count = 0

for name in IDENTITIES:
    identity_filename = os.path.join(OUTPUT_DIR, f"{name.replace(' ', '_')}.jpg")
    if os.path.exists(identity_filename):
        print(f"⚠️  Skipping {name} — image already exists.")
        continue

    print(f"📥 Downloading 1 image for {name}...")
    try:
        crawler = GoogleImageCrawler(storage={"root_dir": OUTPUT_DIR})
        crawler.crawl(keyword=f"{name} face", max_num=1)

        # Rename downloaded image to identity name
        identity_folder = OUTPUT_DIR
        files = [f for f in os.listdir(identity_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if files:
            latest_file = max([os.path.join(identity_folder, f) for f in files], key=os.path.getctime)
            os.rename(latest_file, identity_filename)
            success_count += 1
        else:
            print(f"❌ No image found for {name}")
            fail_count += 1
    except Exception as e:
        print(f"❌ Failed: {name} — {str(e)}")
        fail_count += 1
    time.sleep(1.0)

print(f"\n✅ Done: Downloaded {success_count} identity images.")
print(f"❌ Failed for {fail_count} identities.")
print("➡️ Next: Run detect_and_align.py to align these into data/reference_images/")