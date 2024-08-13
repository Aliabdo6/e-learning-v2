import mongoose from "mongoose";

const PostSchema = new mongoose.Schema({
  title: String,
  content: String,
  author: String,
  createdAt: { type: Date, default: Date.now },
});

export default mongoose.models.Post ||
  mongoose.model("Post", PostSchema);
