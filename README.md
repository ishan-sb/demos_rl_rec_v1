
# Getting the data:

There are two ways:
1) The csv's are from the [Kaggle Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset).  You can download it and unzip it into this folder.  Then, run `python3 prep_data.py` to generate the relevant files needed for the RL run

2) Alternatively, you can download the files from `s3://videoblocks-ml/temp/demos/rl_rec_v1/`.  Ask for help for how to do this via command line!

# Running the recommendation
1) Execute `python run_rl_rec.py`.


# Software setup on Mac OS M1
- Python 3.9
- See next section about the sentence-transformers package
- `pip install inquirer tabulate torch numpy pandas sentence-transformers`

## sentence-transformers package
This package requires a few other things to work, with hdf5 being the most annoying
1. First, get and install [homebrew](https://brew.sh/)
2. Then, get hdf5 with `brew install hdf5`
3. Before running `pip install sentence-transformers`, you need to tell the compiler where to look for relevant hdf5 files (the header and library)
	```
	export CPATH="/opt/homebrew/Cellar/hdf5/1.13.0/include/"
	export LIBRARY_PATH="/opt/homebrew/Cellar/hdf5/1.13.0/lib/"
	pip install sentence-transformers	
	```
4. You may also need to install the following:
	```
	brew install cmake
	brew install rust
	brew install rustup
	```

If you run into blockers doing this, please ask for help!
