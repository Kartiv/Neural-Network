class Neural_Network{

    //NOTES
    // - the default activation function is a sigmoid.

    //TO-DO LIST
    // - add a function to change the activation function

    constructor(structure, thetas = null){

        this.structure = structure;
        
        //create thetas if they do not yet exist
        if(this.thetas){
            this.thetas = thetas;
        }
        else{
            this.thetas = this.genRandThetas();
        }

        this.activationFunction = jsn.sigmoid;
        this.alpha = 10;
    }

    /**
     * 
     * @param {Matrix} input Matrix should be formatted as a row-vector
     */
    forwardProp(input){

        let result = input.T();
        result = new Matrix([[1]].concat(result.data));

        let values = [];

        for(let i=1; i<this.structure.length; i++){
            //calculate linear combination defined by theta coefficients and apply activation function + add bias
            values.push(result);
            result = Neural_Network.addBias(this.thetas[i-1].dot(result).apply(this.activationFunction));
        }

        values.push(result);

        //return output column minus the bias that was added on the last run of the previous loop
        return values;
    }

    backProp(input, target){ //currently the inputs are arrays, might change later

        let values = this.forwardProp(new Matrix([input])); //calculate output
        let y = new Matrix([target]);
        y = y.T();

        //first layer of thetas
        
        let x = values[values.length-2]; //previous layer
        let xm = Neural_Network.removeBias(values[values.length-1]); //current layer

        let gammas = [xm.sub(y).T()];

        for(let i=0; i<this.structure[this.structure.length-1]; i++){

            //i-1 appears where the data structures don't have a bias element to ignore

            let theta_i = new Matrix([this.thetas[this.thetas.length-1].data[i]]);

            let c = this.alpha * 
            ( (xm.data[i][0]-y.data[i][0]) * jsn.dsigmoid(theta_i.dot(x).data[0][0]));

            for(let j=0; j<this.structure[this.structure.length-2]+1; j++){ //+1 since we also have bias

                this.thetas[this.thetas.length-1].data[i][j] -=  x.data[j][0] * c;

            }

        }

        //all of the other layers

        for(let k=this.thetas.length-2; k>-1; k--){

            //calculate the latest gamma
            //this is the vector of the dsigmoid(...) that is direct-producted into the theta matrix
            let pvec = [];
            for(let r=0; r<this.thetas[k+1].data.length; r++){
                pvec.push(this.thetas[k+1].subMatrix(r, r+1, 0, -1).dot(values[k+1]).data[0]);
            }

            pvec = new Matrix(pvec).apply(jsn.dsigmoid);

            //calculate it and add to gammas
            let newGamma = gammas[0].dot( (this.thetas[k+1].subMatrix(0, -1, 1, -1).directProductEveryColumn(pvec)) );
            gammas.splice(0, 0, newGamma);

            //updates thetas
            for(let i=0; i<this.thetas[k].data.length; i++){
                let c = this.alpha * values[k+1].data[i][0] * (1-values[k+1].data[i][0]) *
                gammas[0].data[0][i];
                for(let j=0; j<this.thetas[k].data[0].length; j++){
                    this.thetas[k].data[i][j] -= values[k].data[j] * c;
                }
            }
        }
    }

    evaluate(input){

        input = new Matrix([input]);

        let result = input.T();
        result = new Matrix([[1]].concat(result.data));

        for(let i=1; i<this.structure.length; i++){
            //calculate linear combination defined by theta coefficients and apply activation function + add bias
            result = Neural_Network.addBias(this.thetas[i-1].dot(result).apply(this.activationFunction));
        }

        //return output column minus the bias that was added on the last run of the previous loop
        return Neural_Network.removeBias(result);
    }


    genRandThetas(){

        //goal: array of matrices of the theta_i,j, each entry of the array corresponds to a matrix which corresponds to the theta_i,j of a layer,
        //with theta_i,j being the jth theta of the ith node of the given layer

        //currently thetas are chosen randomly as value between 0 to 1 although this is obviously arbitrary

        let thetas = [];

        for(let i=1; i<this.structure.length; i++){ //adding a layer of thetas

            let layer = [];

            //for each node in the current layer we add its theta array
            for(let j=0; j<this.structure[i]; j++){
                let currentNodeThetas = [];

                //for each node in the previous layer we add a corresponding random theta value
                for(let k=0; k<this.structure[i-1]+1; k++){ //the +1 accounts for a bias node
                    currentNodeThetas.push(Math.random());
                }

                layer.push(currentNodeThetas);

            }

            thetas.push(new Matrix(layer));

        }

        return thetas;
    }

    static addBias(M){
        return new Matrix([[1]].concat(M.data))
    }

    static removeBias(M){
        return new Matrix(M.data.slice(1));
    }

}

let NN = new Neural_Network([2,3,2]);

let data = [[0,0], [1,0], [0,1], [1,1]];
let target = [[0,0], [0,0], [1,1], [1,1]];

for(let i=0; i<100; i++){
    for(let j=0; j<4; j++){
        NN.backProp(data[j], target[j]);
    }
}

for(let i=0; i<4; i++){
    console.log(NN.evaluate(data[i]));
}