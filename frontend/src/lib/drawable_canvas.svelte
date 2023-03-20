<main>
    <div class="canvas-container centered" style="width:{innerWidth*0.9}px; height:{innerHeight*0.9}px; background-color:cadetblue">
      
      <div class="centered" style="width:{canvasSize}px; height:{canvasSize}px">
          <P5 {sketch} />
      </div>
    </div>
  </main>
  
  <svelte:window bind:innerWidth bind:innerHeight />
  
  <script>
    export let width;
    export let height;

    //Remove
    let innerWidth = 0;
    let innerHeight = 0;

    $: width = innerWidth;
    $: height = innerHeight;
    
    //-----

    let verticalBarWidth = 1;
    let horizontalBarHeight = 1;

    let barWidth;
    let barHeight;
    let canvasSize;

    let canvasXCoordinate;
    let canvasYCoordinate;

    let barXCoordinate;
    let barYCoordinate;

    $: if(width<height-horizontalBarHeight){ //Vertical mode
        barWidth = width;
        barHeight = horizontalBarHeight;

        canvasSize = Math.min(width, height);

        canvasXCoordinate = 0;
        canvasYCoordinate = (height-canvasSize)/2.;

        barXCoordinate = 0;
        barYCoordinate = (height-canvasSize)/2 - horizontalBarHeight;
    }
    else if (width < height + verticalBarWidth){ //Transition mode
        barWidth = width;
        barHeight = horizontalBarHeight;

        canvasSize = Math.min(width, height-horizontalBarHeight);

        canvasXCoordinate = (width-canvasSize)/2
        canvasYCoordinate = horizontalBarHeight;

        barXCoordinate = 0;
        barYCoordinate = 0;
    }
    else{ //Landscape mode
        barHeight = height;
        barWidth = verticalBarWidth;

        canvasSize = Math.min(width, height);

        canvasXCoordinate = (width-canvasSize)/2;
        canvasYCoordinate = 0;

        barXCoordinate = (width - canvasSize)/2 + canvasSize;
        barYCoordinate = 0;
    }
    

    import p5Svelte from 'p5-svelte';
    import P5 from 'p5-svelte';
  
    
    $: condition = innerWidth*1.33 <= innerHeight
  
    let circleWidth = 55;
    let circleHeight = 55;
  
    let canvasWidth = 0;
    let canvasHeight = 0;
  
    let bufferWidth = 1000;
    let bufferHeight = 1000;
  
    $: canvasWidth = innerWidth*0.9;
    $: canvasHeight = innerHeight*0.9;
  
    $: canvasSize = Math.min(canvasWidth, canvasHeight);
  
    var drawingBuffer;
  
    const sketch = (p5) => {
      p5.draw = () => {
        if (p5.mouseIsPressed) {
          drawingBuffer.fill(0);
          drawingBuffer.ellipse(p5.mouseX/canvasSize*bufferWidth, p5.mouseY/canvasSize*bufferHeight, 80, 80);
        } 
  
        p5.image(drawingBuffer, 0, 0, canvasSize, canvasSize);
      };
  
  
      p5.setup = () => {
        p5.createCanvas(canvasSize, canvasSize);
        p5.background(200, 0, 100);
        drawingBuffer = p5.createGraphics(bufferWidth, bufferHeight);
        drawingBuffer.background(200, 0, 100);
      };
  
      
  
      p5.windowResized= () => {
        p5.resizeCanvas(canvasSize, canvasSize, true);
      }
    };
  
  
  </script>
  
  
  
  <style>
    .canvas-container{
      position: absolute;
    }
  
    .centered{     
      position: absolute;
      top:0;
      bottom: 0;
      left: 0;
      right: 0;
      
      margin: auto;
    }
  </style>
  
  