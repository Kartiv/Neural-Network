// class Neural_Network{

//     /**
//      * 
//      * @param {array} system layer structure (index 0 is input layer, index 1 is first hidden...)
//      * @param {Matrix} data Training data
//      * @param {Matrix/Vector} target Target Predictions
//      * @param {Int} niter 
//      * @param {float} alpha 
//      * @param {*} thetas 
//      */
//     constructor(system, data, target, niter = 1000, alpha = 1, thetas = null){
//         this.system = system;
//         this.data = data;
//         this.target = target;
//         this.niter = niter;
//         this.alpha = alpha;
//         this.thetas = thetas;
//     }

//     generateThetas(){
//         if(!thetas){
//             var thetas = [];
//             for(let layer=1; layer<this.system.length; layer++){
//                 thetas.push([]);
//                 for(let neuron=0; neuron<this.system[layer]; neuron++){
//                     thetas[thetas.length-1].push(jn.randArr(this.system[layer-1]+1));
//                 }
//             }
//             this.thetas = thetas;
//         }
//     }

//     forwardProp(x0){
//         let t = [];
//         for(let i=0; i<x0.length; i++){
//             t.push(x0[i]);
//         }
//         t.splice(0,0,1);
//         var values = [t];
//         for(let layer=1; layer<this.system.length; layer++){
//             let temp = jn.lstSigmoid(jn.operator(this.thetas[layer-1], values[layer-1]));
//             temp.splice(0,0,1);
//             values.push(temp);
//         }
//         return values
//     }

//     backProp(){
//         for(let point=0; point<this.data.length; point++){ //for each data point
//             let values = this.forwardProp(this.data[point]); //calculate all of the values
//             let deltas = [[]]; //first (last) layer of deltas
//             for(let i=1; i<values[values.length-1].length; i++){ //ignore bias which is the first element
//                 deltas[0].push(-2*(this.target[point][i-1]-values[values.length-1][i])*values[values.length-1][i]*(1-values[values.length-1][i])) //definition of deltas
//                 for(let j=0; j<values[values.length-2].length; j++){
//                     this.thetas[this.thetas.length-1][i-1][j]-=this.alpha * deltas[0][deltas[0].length-1]*values[values.length-2][j] //gradient descent
//                 }
//             }

//             for(let layer=values.length-2; layer>0; layer--){ //dont include first layer because we already calculated it, dont include last because its the input layer
//                 deltas.push([]);
//                 for(let i=1; i<values[layer].length; i++){ //ignore bias, doesnt matter for backwards propagation
//                     let s = 0; //calulate dot product by hand because numpy can go suck a fucking lollipop. this is the dot product/sum in definition for deltas
//                     for(let k=1; k<values[layer+1].length; k++){ //once again, fuck the bias
//                         s+=deltas[deltas.length-2][k-1] * this.thetas[layer][k-1][i] //deltas doesnt include a bias (starts from index 1), so we k-1 instead of k. additionaly, thetas array is made so thetas[layer]
//                     } //are the thetas for the next layer, none go to bias, 2d arr which is practically a single dim array so we take 0 coord, ith neuron is what we are looking for
//                     deltas[deltas.length-1].push(s*values[layer][i]*(1-values[layer][i])); 
//                     for(let j=0; j<values[layer-1].length; j++){//adjust all of the thetas
//                         this.thetas[layer-1][i-1][j] -= this.alpha * deltas[deltas.length-1][deltas[deltas.length-1].length-1]*values[layer-1][j]; //read
//                     }
//                 }
//             }
//         }
//     }

//     solve(){
//         this.generateThetas();
//         for(let run=0; run<this.niter; run++){
//             this.backProp();
//         }        
//     }
    

//     evaluate(){
//         let count = 0;
//         for(let i=0; i<this.data.length; i++){
//             if (jn.areEqual(jn.roundArr(this.predict(this.data[i])), this.target[i])){
//                 count++;
//             }
//         }
//         return {"count": count, "successRate": count/this.data.length}
//     }

//     predict(x0){
//         let temp = this.forwardProp(x0)
//         temp = temp[temp.length-1];
//         temp.splice(0,1);
//         return temp;
//     }
// }

class Neural_Network{

    //NOTES
    // - the default activation function is a sigmoid.

    //TO-DO LIST
    // - include a function that automatically adds a bias node to each layer
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
        this.errorFunction = (output, target) => {let v = target.sub(output); return v.T().dot(v)[0]};
    }

    /**
     * 
     * @param {Matrix} input Matrix should be formatted as a row-vector
     */
    forwardProp(input){

        let result = input.T();

        for(let i=1; i<this.structure.length; i++){
            result = this.thetas[i-1].dot(result).apply(this.activationFunction); //calculate linear combination defined by theta coefficients and apply activation function
        }

        return result;
    }

    backProp(input, target){
        https://towardsdatascience.com/everything-you-need-to-know-about-neural-networks-and-backpropagation-machine-learning-made-easy-e5285bc2be3a
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
                for(let k=0; k<this.structure[i-1]; k++){
                    currentNodeThetas.push(Math.random());
                }

                layer.push(currentNodeThetas);

            }

            thetas.push(new Matrix(layer));

        }

        return thetas;
    }

}
