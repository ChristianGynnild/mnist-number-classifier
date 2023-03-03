# mnist-number-classifier
An AI number classifier trained on the MNIST handwritten digit database.

The dataset can be found [here](http://yann.lecun.com/exdb/mnist/).

the project uses wasm-pack to build the wasm file. Run the following command in a shell to build the project: `wasm-pack build --target web --out-dir ./static/pkg`

While the devolopment server can be run using `flask --app server run`