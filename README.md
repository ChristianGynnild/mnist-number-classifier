# mnist-number-classifier
An AI number classifier trained on the MNIST handwritten digit database.

The dataset can be found [here](http://yann.lecun.com/exdb/mnist/).


## Devolopment servers
First install node-JS and Python.

Setup a suitable Python enviroment and then install the dependencies in requirements.txt.

To start the devolopment servers run `flask --app server run --debug --port=5000` inside the root of the project structure. 

Then run `npm run dev` inside the frontend folder and open http://localhost:8080/static/ inside the browser.