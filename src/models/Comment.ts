import mongoose from "mongoose";

const CommentSchema = new mongoose.Schema({
  content: String,
  author: String,
  postId: mongoose.Schema.Types.ObjectId,
  createdAt: { type: Date, default: Date.now },
});

export default mongoose.models.Comment ||
  mongoose.model("Comment", CommentSchema);
