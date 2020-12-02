const fs = require('fs');
const http = require('http');

// dataset parameters; dataset file assumes all numeric inputs with no missing values,
// in which each line is a single sample with all features followed by all outputs, delimited by commas
// (standard CSV format with no header row)
const dataset_file = 'http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data';
const dataset_name = 'spam';

// network parameters
const hidden_nodes = 64;
const output_nodes = 1;

const parse_dataset = dataset_string => {
	// put into 2d array
	const data = dataset_string.split('\n').map(row => row.split(',').map(str => parseFloat(str)));

	// may need to remove last row (is a problem in spam dataset)
	if (data[data.length-1].length != data[0].length) {
		data.splice(data.length-1);
	}

	// shuffle dataset using fisher-yates
	const N = data.length;
	for (let i = 0; i < N; ++i) {
		const idx = i+Math.floor((N-i)*Math.random());
		const tmp = [...data[idx]];
		data[idx] = [...data[i]];
		data[i] = tmp;
	}

	// separate features and labels
	const X = data.map(row => row.slice(0, -output_nodes));
	const Y = data.map(row => row.slice(-output_nodes));

	// minmaxscalar
	const col_maxes = X.reduce((acc, row, idx) => acc.map((val, i) => Math.max(val, row[i])));
	const X_scaled = X.map(row => row.map((val, i) => val / col_maxes[i]));

	// split into train and test datasets
	const split = Math.floor(N*0.8);
	const X_train = X_scaled.slice(0, split);
	const Y_train = Y.slice(0, split);
	const X_test = X_scaled.slice(split);
	const Y_test = Y.slice(split);
	const train_dataset = X_train.map((row, i) => row.concat(Y_train[i]));
	const test_dataset = X_test.map((row, i) => row.concat(Y_test[i]));

	// N = number of samples
	// F = number of features
	// O = number of outputs (output nodes)
	// H = number of hidden nodes
	const N_train = split;
	const N_test = N-split;
	const F = X[0].length;
	const O = output_nodes;	// global var
	const H = hidden_nodes;	// global var

	// export datasets
	const train_outfile = `${dataset_name}.train`;
	const test_outfile = `${dataset_name}.test`;

	const train_dataset_string = train_dataset.map(row => row.map((num, i) => num.toFixed(i == F ? 0 : 3)).join(' ')).join('\n');
	const test_dataset_string = test_dataset.map(row => row.map((num, i) => num.toFixed(i == F ? 0 : 3)).join(' ')).join('\n');

	const train_string = `${N_train} ${F} ${O}\n${train_dataset_string}`;
	const test_string = `${N_test} ${F} ${O}\n${test_dataset_string}`;

	fs.writeFileSync(train_outfile, train_string);
	fs.writeFileSync(test_outfile, test_string);

	// Standard Normal variate using Box-Muller transform.
	// from: https://stackoverflow.com/a/36481059/2397327
	function randn_bm() {
	    var u = 0, v = 0;
	    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
	    while(v === 0) v = Math.random();
	    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
	}

	// generate random network description
	const first_layer_weights = [...Array(H)].map(_ => [...Array(F+1)].map(_ => randn_bm()));
	const second_layer_weights = [...Array(O)].map(_ => [...Array(H+1)].map(_ => randn_bm()));

	const first_layer_weights_string = first_layer_weights.map(node => node.map(num => num.toFixed(3)).join(' ')).join('\n');
	const second_layer_weights_string = second_layer_weights.map(node => node.map(num => num.toFixed(3)).join(' ')).join('\n');

	const weights_string = `${F} ${H} ${O}\n${first_layer_weights_string}\n${second_layer_weights_string}`;
	const weights_outfile = `${dataset_name}.init`;

	fs.writeFileSync(weights_outfile, weights_string);
};

// initiate get and parse dataset
http.get(dataset_file, res => {
	res.statusCode !== 200 && console.error(error);

	let res_body = '';
	res.on('data', chunk => res_body += chunk);
	res.on('end', () => parse_dataset(res_body));

}).on('error', console.error);
